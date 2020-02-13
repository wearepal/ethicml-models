"""Wrapper for calling the fair GP model."""
from pathlib import Path
from tempfile import TemporaryDirectory

from typing import List, Optional, Dict, Any, ClassVar, Union, Mapping, Tuple, Callable, Sequence
from typing_extensions import Literal
import numpy as np
import pandas as pd

from ethicml.common import implements
from ethicml.utility import DataTuple, TestTuple, Prediction
from ethicml.algorithms.inprocess import InAlgorithmAsync, InstalledModel

from .common import ROOT_PATH

__all__ = ["GPyT", "GPyTDemPar", "GPyTEqOdds"]

PRED_FNAME = "predictions.npz"
MAX_EPOCHS = 1000
MAX_BATCH_SIZE = 10100  # can go up to 10000
MAX_NUM_INDUCING = 5000  # 2500 seems to be more than enough
SEED = 888
CODE_DIR = ROOT_PATH / "implementations" / "fairgp"

FlagType = Dict[str, Union[int, float, str, bool]]


class GPyT(InstalledModel):
    """Normal GP model."""

    basename: ClassVar[str] = "GPyT"

    def __init__(
        self,
        s_as_input: bool = True,
        flags: Optional[FlagType] = None,
        code_dir: Path = CODE_DIR,
        name: Optional[str] = None,
    ):
        """Instantiate the model.

        Args:
            s_as_input: if True, use s as an input feature
            flags: (optional) a dictionary that can overwrite *any* flag
            code_dir: (optional) a directory where the GPyT code can be found. if this is not given,
                      then the code is downloaded from GitHub
        """
        if name is None:
            name = f"{self.basename}_in_{s_as_input}"
        super().__init__(name=name, dir_name=str(code_dir), top_dir=".")
        self.s_as_input = s_as_input
        self.flag_overwrites: Dict[str, Any] = {} if flags is None else flags

    @implements(InAlgorithmAsync)
    async def run_async(self, train: DataTuple, test: TestTuple) -> Prediction:
        (ytrain,), label_converter = _fix_labels([train.y.to_numpy()])
        raw_data = dict(
            xtrain=train.x.to_numpy(),
            strain=train.s.to_numpy(),
            ytrain=ytrain,
            xtest=test.x.to_numpy(),
            stest=test.s.to_numpy(),
            # ytest=ytest,
        )
        parameters = self._additional_parameters(raw_data)

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "data.npz"
            np.savez(data_path, **raw_data)
            model_name = "local"
            flags = _flags(
                parameters,
                str(data_path),
                tmpdir,
                self.s_as_input,
                model_name,
                raw_data["ytrain"].shape[0],
            )
            await self._run_gpyt(flags)

            # Read the results from the numpy file 'predictions.npz'
            with (tmp_path / model_name / PRED_FNAME).open("rb") as file_obj:
                output = np.load(file_obj)
                pred_mean = output["pred_mean"]

        predictions = label_converter((pred_mean > 0.5).astype(raw_data["ytrain"].dtype)[:, 0])
        return Prediction(hard=pd.Series(predictions))

    async def _run_gpyt(self, flags: FlagType) -> None:
        """Generate command to run GPyT."""
        cmd = [str(self._code_path / "run_fairgp.py")]
        # apply flag overwrites
        cmd_flags = {**flags, **self.flag_overwrites}
        for key, value in cmd_flags.items():
            cmd += [f"--{key}", str(value)]
        await self._call_script(cmd)

    @staticmethod
    def _additional_parameters(raw_data: Mapping[str, np.ndarray]) -> FlagType:
        del raw_data
        return dict(lik="BaselineLikelihood")


class GPyTDemPar(GPyT):
    """GP algorithm which enforces demographic parity."""

    MEAN: ClassVar[Literal[2]] = 2  # pylint: disable=invalid-name
    MIN: ClassVar[Literal[3]] = 3  # pylint: disable=invalid-name
    MAX: ClassVar[Literal[4]] = 4  # pylint: disable=invalid-name

    def __init__(
        self,
        target_acceptance: Union[None, float, Tuple[float, float]] = None,
        average_prediction: bool = False,
        target_mode: Literal[2, 3, 4] = MEAN,
        marginal: bool = False,
        precision_target: float = 1.0,
        s_as_input: bool = True,
        flags: Optional[FlagType] = None,
        code_dir: Path = CODE_DIR,
    ):
        """Instantiate the model.

        Args:
            s_as_input: should the sensitive attribute be part of the input?
            target_acceptance: which acceptance rate to target
            average_prediction: whether to use to average of all possible sensitive attributes for
                                predictions
            target_mode: if no target rate is given, how is the target chosen?
            marginal: when doing average_prediction, should the prior of s be taken into account?
            precision_target: how similar should target labels and true labels be
        """
        if s_as_input and average_prediction:
            name = f"{self.basename}_dem_par_av_True"
            if marginal:
                name += "_marg"
        else:
            name = f"{self.basename}_dem_par_in_{s_as_input}"
        if target_acceptance is not None:
            name += f"_tar_{target_acceptance}"
        elif target_mode != self.MEAN:
            if target_mode == self.MIN:
                name += "_tar_min"
            elif target_mode == self.MAX:
                name += "_tar_max"
            else:
                name += f"_tar_{target_mode}"
        if precision_target != 1.0:
            name += f"_pt_{precision_target}"
        self.target_acceptance = target_acceptance
        self.target_mode = target_mode
        self.average_prediction = average_prediction
        self.marginal = marginal
        self.precision_target = precision_target
        super().__init__(name=name, s_as_input=s_as_input, flags=flags, code_dir=code_dir)

    def _additional_parameters(self, raw_data: Mapping[str, np.ndarray]) -> FlagType:
        biased_acceptance = compute_bias(raw_data["ytrain"], raw_data["strain"])

        target_rate: Union[float, Tuple[float, float]]
        if self.target_acceptance is None:
            if self.target_mode == self.MEAN:
                target_rate = 0.5 * (biased_acceptance[0] + biased_acceptance[1])
            elif self.target_mode == self.MIN:
                target_rate = min(biased_acceptance[0], biased_acceptance[1])
            elif self.target_mode == self.MAX:
                target_rate = max(biased_acceptance[0], biased_acceptance[1])
            else:
                acc_min = min(biased_acceptance[0], biased_acceptance[1])
                acc_max = max(biased_acceptance[0], biased_acceptance[1])
                target_rate = acc_min + self.target_mode * (acc_max - acc_min)
        else:
            target_rate = self.target_acceptance

        if self.marginal:
            p_s = prior_s(raw_data["strain"])
        else:
            p_s = (0.5, 0.5)

        return dict(
            lik="TunePrLikelihood",
            target_rate1=target_rate[0] if isinstance(target_rate, tuple) else target_rate,
            target_rate2=target_rate[1] if isinstance(target_rate, tuple) else target_rate,
            biased_acceptance1=biased_acceptance[0],
            biased_acceptance2=biased_acceptance[1],
            probs_from_flipped=False,
            average_prediction=self.average_prediction,
            p_s0=p_s[0],
            p_s1=p_s[1],
            p_ybary0_or_ybary1_s0=self.precision_target,
            p_ybary0_or_ybary1_s1=self.precision_target,
        )


class GPyTEqOdds(GPyT):
    """GP algorithm which enforces equality of opportunity."""

    def __init__(
        self,
        average_prediction: bool = False,
        tpr: Optional[float] = None,
        marginal: bool = False,
        tnr0: Optional[float] = None,
        tnr1: Optional[float] = None,
        tpr0: Optional[float] = None,
        tpr1: Optional[float] = None,
        s_as_input: bool = True,
        flags: Optional[FlagType] = None,
        code_dir: Path = CODE_DIR,
    ):
        """Init GP with eq. odds."""
        if self.s_as_input and average_prediction:
            name = "{self.basename}_eq_odds_av_True"
            if marginal:
                name += "_marg"
        else:
            name = f"{self.basename}_eq_odds_in_{self.s_as_input}"

        self.odds: Optional[Dict[str, float]] = None
        if any(x is not None for x in [tnr0, tnr1, tpr0, tpr1]):  # if any of them is not `None`
            self.odds = {}
            for val, name, target in [
                (tnr0, "0tnr", "p_ybary0_s0"),
                (tnr1, "1tnr", "p_ybary0_s1"),
                (tpr0, "0tpr", "p_ybary1_s0"),
                (tpr1, "1tpr", "p_ybary1_s1"),
            ]:
                if val is not None:
                    self.odds[target] = val
                    name += f"_{name}_{val}"  # add to name
                else:
                    self.odds[target] = 1.0  # default value
        elif tpr is not None:
            self.odds = dict(p_ybary0_s0=1.0, p_ybary0_s1=1.0, p_ybary1_s0=tpr, p_ybary1_s1=tpr)
            name += f"_tpr_{tpr}"

        self.average_prediction = average_prediction
        self.marginal = marginal
        super().__init__(name=name, s_as_input=s_as_input, flags=flags, code_dir=code_dir)

    def _additional_parameters(self, raw_data: Mapping[str, np.ndarray]) -> FlagType:
        biased_acceptance = compute_bias(raw_data["ytrain"], raw_data["strain"])

        if self.marginal:
            p_s = prior_s(raw_data["strain"])
        else:
            p_s = (0.5, 0.5)

        return dict(
            lik="TuneTprLikelihood",
            p_ybary0_s0=1.0,
            p_ybary0_s1=1.0,
            p_ybary1_s0=1.0,
            p_ybary1_s1=1.0,
            biased_acceptance1=biased_acceptance[0],
            biased_acceptance2=biased_acceptance[1],
            average_prediction=self.average_prediction,
            p_s0=p_s[0],
            p_s1=p_s[1],
        )

    @implements(InAlgorithmAsync)
    async def run_async(self, train: DataTuple, test: TestTuple) -> Prediction:
        (ytrain,), label_converter = _fix_labels([train.y.to_numpy()])
        raw_data = dict(
            xtrain=train.x.to_numpy(),
            strain=train.s.to_numpy(),
            ytrain=ytrain,
            xtest=test.x.to_numpy(),
            stest=test.s.to_numpy(),
            # ytest=ytest,
        )
        parameters = self._additional_parameters(raw_data)

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / Path("data.npz")
            model_name = "local"  # f"run{self.counter}_s_as_input_{self.s_as_input}"
            flags = _flags(
                parameters,
                str(data_path),
                tmpdir,
                self.s_as_input,
                model_name,
                len(raw_data["ytrain"]),
            )

            if self.odds is None:
                # Split the training data into train and dev and save it to `data.npz`
                train_dev_data = split_train_dev(
                    raw_data["xtrain"], raw_data["ytrain"], raw_data["strain"]
                )
                np.savez(data_path, **train_dev_data)

                # First run
                await self._run_gpyt(flags)

                # Read the results from the numpy file 'predictions.npz'
                with (tmp_path / model_name / PRED_FNAME).open("rb") as file_obj:
                    output = np.load(file_obj)
                    prediction_on_train = output["pred_mean"]
                preds = (prediction_on_train > 0.5).astype(np.int32)
                odds = compute_odds(train_dev_data["ytest"], preds, train_dev_data["stest"])

                # Enforce equality of opportunity
                opportunity = min(odds["p_ybary1_s0"], odds["p_ybary1_s1"])
                odds["p_ybary1_s0"] = opportunity
                odds["p_ybary1_s1"] = opportunity
                flags.update({"epochs": 2 * flags["epochs"], **odds})
            else:
                flags.update(self.odds)

            # Save with real test data
            np.savez(data_path, **raw_data)

            # Second run
            await self._run_gpyt(flags)

            # Read the results from the numpy file 'predictions.npz'
            with (tmp_path / model_name / PRED_FNAME).open("rb") as file_obj:
                output = np.load(file_obj)
                pred_mean = output["pred_mean"]

        # Convert the result to the expected format
        predictions = label_converter((pred_mean > 0.5).astype(raw_data["ytrain"].dtype)[:, 0])
        return Prediction(hard=pd.Series(predictions))

    @property
    def name(self) -> str:
        """Getter for algorithm name."""
        return self.__name


def prior_s(sensitive: np.ndarray) -> Tuple[float, float]:
    """Compute the bias in the labels with respect to the sensitive attributes."""
    return (np.sum(sensitive == 0) / len(sensitive), np.sum(sensitive == 1) / len(sensitive))


def compute_bias(labels: np.ndarray, sensitive: np.ndarray) -> Tuple[float, float]:
    """Compute the bias in the labels with respect to the sensitive attributes."""
    rate_y1_s0 = np.sum(labels[sensitive == 0] == 1) / np.sum(sensitive == 0)
    rate_y1_s1 = np.sum(labels[sensitive == 1] == 1) / np.sum(sensitive == 1)
    return rate_y1_s0, rate_y1_s1


def compute_odds(
    labels: np.ndarray, predictions: np.ndarray, sensitive: np.ndarray
) -> Dict[str, float]:
    """Compute the bias in the predictions with respect to the sensitive attr. and the labels."""
    return dict(
        p_ybary0_s0=np.mean(predictions[(labels == 0) & (sensitive == 0)] == 0),
        p_ybary1_s0=np.mean(predictions[(labels == 1) & (sensitive == 0)] == 1),
        p_ybary0_s1=np.mean(predictions[(labels == 0) & (sensitive == 1)] == 0),
        p_ybary1_s1=np.mean(predictions[(labels == 1) & (sensitive == 1)] == 1),
    )


def _fix_labels(
    labels: Sequence[np.ndarray],
) -> Tuple[List[np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Make sure that labels are either 0 or 1.

    Args"
        labels: the labels as a list of numpy arrays
        positive_class_val: the value that corresponds to a "positive" predictions

    Returns:
        the fixed labels and a function to convert the fixed labels back to the original format
    """
    label_values = list(np.unique(labels[0]))
    if label_values == [0, 1]:

        def _do_nothing(inp: np.ndarray) -> np.ndarray:
            return inp

        return list(labels), _do_nothing
    if label_values == [-1, 1]:

        def _converter(label: np.ndarray) -> np.ndarray:
            return (2 * label) - 1

        return [(y + 1) / 2 for y in labels], _converter
    raise ValueError("Labels have unknown structure")


def split_train_dev(
    inputs: np.ndarray, labels: np.ndarray, sensitive: np.ndarray
) -> Dict[str, np.ndarray]:
    """Split the given data into train and dev set with the proportion of labels being preserved."""
    n_total = inputs.shape[0]
    idx_s0_y0 = ((sensitive == 0) & (labels == 0)).nonzero()[0]
    idx_s0_y1 = ((sensitive == 0) & (labels == 1)).nonzero()[0]
    idx_s1_y0 = ((sensitive == 1) & (labels == 0)).nonzero()[0]
    idx_s1_y1 = ((sensitive == 1) & (labels == 1)).nonzero()[0]

    train_fraction: List[np.ndarray[np.int64]] = []
    test_fraction: List[np.ndarray[np.int64]] = []
    for idx in [idx_s0_y0, idx_s0_y1, idx_s1_y0, idx_s1_y1]:
        np.random.shuffle(idx)

        split_idx = int(len(idx) * 0.5) + 1  # make sure the train part is at least half
        train_fraction_a = idx[:split_idx]
        test_fraction_a = idx[split_idx:]
        train_fraction += [train_fraction_a]
        test_fraction += [test_fraction_a]
    train_idx = np.concatenate(train_fraction, axis=0)
    test_idx = np.concatenate(test_fraction, axis=0)
    xtrain, ytrain, strain = (
        inputs[train_idx],
        labels[train_idx],
        sensitive[train_idx],
    )
    # ensure that the train set has exactly the same size as the given set
    # (otherwise inducing inputs has wrong shape)
    return dict(
        xtrain=np.concatenate((xtrain, xtrain))[:n_total],
        ytrain=np.concatenate((ytrain, ytrain))[:n_total],
        strain=np.concatenate((strain, strain))[:n_total],
        xtest=inputs[test_idx],
        ytest=labels[test_idx],
        stest=sensitive[test_idx],
    )


def _flags(
    parameters: FlagType,
    data_path: str,
    save_dir: str,
    s_as_input: bool,
    model_name: str,
    num_train: int,
) -> FlagType:
    batch_size = min(MAX_BATCH_SIZE, num_train)
    # epochs = _num_epochs(num_train)
    epochs = 70
    return {
        **dict(
            inf="Variational",
            data="sensitive_from_numpy",
            dataset_path=data_path,
            cov="RBFKernel",
            mean="ZeroMean",
            optimizer="Adam",
            lr=0.05,
            # lr=0.1,
            model_name=model_name,
            batch_size=batch_size,
            # epochs=min(MAX_EPOCHS, _num_epochs(num_train)),
            epochs=min(epochs, MAX_EPOCHS),
            eval_epochs=100000,  # we can unfortunately not run evaluations because of no y_test
            summary_steps=100000,
            chkpt_epochs=100000,
            save_dir=save_dir,  # "/home/ubuntu/out2/",
            plot="",
            logging_steps=1,
            gpu=0,
            preds_path=PRED_FNAME,  # save the predictions into `predictions.npz`
            num_samples=20,
            optimize_inducing=True,
            length_scale=1.2,
            sf=1.0,
            iso=False,
            num_samples_pred=2000,
            s_as_input=s_as_input,
            # num_inducing=MAX_NUM_INDUCING,
            num_inducing=_num_inducing(num_train),
            manual_seed=SEED,
            # metrics=(
            #     "binary_accuracy,pred_rate_y1_s0,pred_rate_y1_s1,base_rate_y1_s0,base_rate_y1_s1,"
            #     "pred_odds_yhaty1_s0,pred_odds_yhaty1_s1,pred_odds_yhaty0_s0,pred_odds_yhaty0_s1"
            # ),
        ),
        **parameters,
    }


def _num_inducing(num_train: int) -> int:
    """Adaptive number of inducing inputs.

    num_train == 4,000 => num_inducing == 1121
    num_train == 20,000 => num_inducing == 2507
    """
    return int(2500 / 141 * np.sqrt(num_train))


def _num_epochs(num_train: int) -> int:
    """Adaptive number of epochs.

    num_train == 4,000 => num_epochs == 125.7
    num_train == 20,000 => num_epochs == 84
    """
    return int(1000 / np.power(num_train, 1 / 4))
