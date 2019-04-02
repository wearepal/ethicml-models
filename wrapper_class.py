"""Code for calling fair-gpytorch"""
from pathlib import Path
from tempfile import TemporaryDirectory
from subprocess import call
import json
import numpy as np

# TODO: find a better way to specify the path
GPYT_PATH = "/home/ubuntu/code/fair-gpytorch/run.py"
# PYTHON_EXE = "/home/ubuntu/anaconda3/envs/pytorch_p36/bin/python -m pdb -c continue"
PYTHON_EXE = "/home/ubuntu/anaconda3/envs/pytorch_p36/bin/python -u"
MAX_EPOCHS = 1000
MAX_BATCH_SIZE = 10100  # can go up to 10000
MAX_NUM_INDUCING = 5000  # 2500 seems to be more than enough
SEED = 1234


class GPyT:
    """
    This class calls the fair-gpytorch code
    """
    basename = "GPyT"

    def __init__(self, s_as_input=True):
        super().__init__()
        self.counter = 0
        self.s_as_input = s_as_input
        self.name = f"{self.basename}_in_{s_as_input}"

    def run(self, *data):
        """
        Runs the algorithm and returns the predicted classifications on the test set.  The given
        train and test data still contains the sensitive_attrs. This run of the algorithm should
        focus on the single given sensitive attribute.

        Be sure that the returned predicted classifications are of the same type as the class
        attribute in the given test_df. If this is not the case, some metric analyses may fail to
        appropriately compare the returned predictions to their desired values.

        Args:
            train_df: Pandas datafram with the training data
            test_df: Pandas datafram with the test data
            class_attr: string that names the column with the label
            positive_class_val: the value for the label which is considered the positive class
                (usually '1')
            sensitive_attrs: list of all available sensitive attributes (all but one should be
                ignored)
            single_sensitive: name of the sensitive attribute that is considered in this run
            privileged_vals: the groups that are considered privileged (usually '1')
            params: a dictionary mapping from algorithm-specific parameter names to the desired
                values. If the implementation of run uses different values, these should be modified
                in the params dictionary as a way of returning the used values to the caller.
        """
        self.counter += 1
        # Separate the data and make sure the labels are either 0 or 1
        raw_data, label_converter, gpu = prepare_data(*data)

        # Set algorithm dependent parameters
        parameters = self._additional_parameters(raw_data)

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Save the data in a numpy file called 'data.npz'
            data_path = tmp_path / Path("data.npz")
            np.savez(data_path, **raw_data)

            # Construct and execute command
            model_name = "local"  # f"run{self.counter}_s_as_input_{self.s_as_input}"
            self.run_gpyt(_flags(parameters, str(data_path), tmpdir, self.s_as_input, model_name,
                                 raw_data['ytrain'].shape[0], gpu))

            # Read the results from the numpy file 'predictions.npz'
            with (tmp_path / model_name / "predictions.npz").open('rb') as f:
                output = np.load(f)
                pred_mean = output['pred_mean']

        # Convert the result to the expected format
        return label_converter((pred_mean > 0.5).astype(raw_data['ytest'].dtype)[:, 0]), []

    @staticmethod
    def run_gpyt(flags):
        """Run UniversalGP as a separte process"""
        cmd = f"{PYTHON_EXE} {GPYT_PATH} "
        for key, value in flags.items():
            if isinstance(value, str):
                cmd += f" --{key}=\"{value}\""
            else:
                cmd += f" --{key}={value}"
        call(cmd, shell=True)  # run `cmd`

    @staticmethod
    def get_param_info():
        """
        Returns a dictionary mapping algorithm parameter names to a list of parameter values to be
        explored. This function should only be implemented if the algorithm has specific parameters
        that should be tuned, e.g., for trading off between fairness and accuracy.
        """
        return {}

    @staticmethod
    def get_supported_data_types():

        return set(["numerical-binsensitive"])

    def get_name(self):
        """
        Returns the name for the algorithm. This must be a unique name, so it is suggested that this
        name is simply <firstauthor>. If there are mutliple algorithms by the same author(s), a
        suggested modification is <firstauthor-algname>. This name will appear in the resulting CSVs
        and graphs created when performing benchmarks and analysis.
        """
        return self.name

    def get_default_params(self):
        """
        Returns a dictionary mapping from parameter names to default values that should be used with
        the algorithm. If not implemented by a specific algorithm, this returns the empty
        dictionary.
        """
        return {}

    @staticmethod
    def _additional_parameters(_):
        return dict(
            lik='BaselineLikelihood',
        )

    def _save_in_json(self, save_path):
        """Save the settings in a JSON file called 'settings.json'"""
        with open(save_path / Path("settings.json"), 'w') as f:
            data = dict(s_as_input=self.s_as_input, counter=self.counter)
            json.dump(data, f, ensure_ascii=False, indent=2)


class GPyTDemPar(GPyT):
    """GP algorithm which enforces demographic parity"""
    MEAN = 2
    MIN = 3
    MAX = 4

    def __init__(self, s_as_input=True, target_acceptance=None, average_prediction=False,
                 target_mode=MEAN, marginal=False, precision_target=1.0):
        """
        Args:
            s_as_input: should the sensitive attribute be part of the input?
            target_acceptance: which acceptance rate to target
            average_prediction: whether to use to average of all possible sensitive attributes for
                                predictions
            target_mode: if no target rate is given, how is the target chosen?
            marginal: when doing average_prediction, should the prior of s be taken into account?
            precision_target: how similar should target labels and true labels be
        """
        super().__init__(s_as_input=s_as_input)
        if s_as_input and average_prediction:
            self.name = f"{self.basename}_dem_par_av_True"
            if marginal:
                self.name += "_marg"
        else:
            self.name = f"{self.basename}_dem_par_in_{s_as_input}"
        if target_acceptance is not None:
            self.name += f"_tar_{target_acceptance}"
        elif target_mode != self.MEAN:
            if target_mode == self.MIN:
                self.name += "_tar_min"
            elif target_mode == self.MAX:
                self.name += "_tar_max"
            else:
                self.name += f"_tar_{target_mode}"
        if precision_target != 1.0:
            self.name += f"_pt_{precision_target}"
        self.target_acceptance = target_acceptance
        self.target_mode = target_mode
        self.average_prediction = average_prediction
        self.marginal = marginal
        self.precision_target = precision_target

    def _additional_parameters(self, raw_data):
        biased_acceptance = compute_bias(raw_data['ytrain'], raw_data['strain'])

        if self.target_acceptance is None:
            if self.target_mode == self.MEAN:
                target_rate = .5 * (biased_acceptance[0] + biased_acceptance[1])
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
            p_s = prior_s(raw_data['strain'])
        else:
            p_s = [0.5] * 2

        return dict(
            lik='TunePrLikelihood',
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
    """GP algorithm which enforces equality of opportunity"""
    def __init__(self, s_as_input=True, average_prediction=False, tpr=None, marginal=False,
                 tnr0=None, tnr1=None, tpr0=None, tpr1=None):
        super().__init__(s_as_input=s_as_input)
        if s_as_input and average_prediction:
            self.name = "{self.basename}_eq_opp_av_True"
            if marginal:
                self.name += "_marg"
        else:
            self.name = f"{self.basename}_eq_opp_in_{s_as_input}"

        self.odds = None
        if any(x is not None for x in [tnr0, tnr1, tpr0, tpr1]):  # if any of them is not `None`
            self.odds = {}
            for val, name, target in [(tnr0, '0tnr', 'p_ybary0_s0'), (tnr1, '1tnr', 'p_ybary0_s1'),
                                      (tpr0, '0tpr', 'p_ybary1_s0'), (tpr1, '1tpr', 'p_ybary1_s1')]:
                if val is not None:
                    self.odds[target] = val
                    self.name += f"_{name}_{val}"  # add to name
                else:
                    self.odds[target] = 1.0  # default value
        elif tpr is not None:
            self.odds = dict(
                p_ybary0_s0=1.0,
                p_ybary0_s1=1.0,
                p_ybary1_s0=tpr,
                p_ybary1_s1=tpr,
            )
            self.name += f"_tpr_{tpr}"

        self.average_prediction = average_prediction
        self.marginal = marginal

    def _additional_parameters(self, raw_data):
        biased_acceptance = compute_bias(raw_data['ytrain'], raw_data['strain'])

        if self.marginal:
            p_s = prior_s(raw_data['strain'])
        else:
            p_s = [0.5] * 2

        return dict(
            lik='TuneTprLikelihood',
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

    def run(self, *data):
        self.counter += 1
        raw_data, label_converter, gpu = prepare_data(*data)

        parameters = self._additional_parameters(raw_data)

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / Path("data.npz")
            model_name = "local"  # f"run{self.counter}_s_as_input_{self.s_as_input}"
            flags = _flags(parameters, str(data_path), tmpdir, self.s_as_input, model_name,
                           len(raw_data['ytrain']), gpu)

            if self.odds is None:
                # Split the training data into train and dev and save it to `data.npz`
                train_dev_data = split_train_dev(
                    raw_data['xtrain'], raw_data['ytrain'], raw_data['strain'])
                np.savez(data_path, **train_dev_data)

                # First run
                self.run_gpyt(flags)

                # Read the results from the numpy file 'predictions.npz'
                prediction_on_train = np.load(tmp_path / Path(model_name) / Path("predictions.npz"))
                preds = (prediction_on_train['pred_mean'] > 0.5).astype(int)
                odds = compute_odds(train_dev_data['ytest'], preds, train_dev_data['stest'])

                # Enforce equality of opportunity
                opportunity = min(odds['p_ybary1_s0'], odds['p_ybary1_s1'])
                odds['p_ybary1_s0'] = opportunity
                odds['p_ybary1_s1'] = opportunity
                flags.update({'train_steps': 2 * flags['train_steps'], **odds})
            else:
                flags.update(self.odds)

            # Save with real test data
            np.savez(data_path, **raw_data)

            # Second run
            self.run_gpyt(flags)

            # Read the results from the numpy file 'predictions.npz'
            output = np.load(tmp_path / Path(model_name) / Path("predictions.npz"))
            pred_mean = output['pred_mean']

        # Convert the result to the expected format
        return label_converter((pred_mean > 0.5).astype(raw_data['ytest'].dtype)[:, 0]), []


class GPyTCal(GPyT):
    """GP algorithm which enforces calibration"""
    def __init__(self, s_as_input=True, average_prediction=False, npv0=1.0, npv1=1.0, ppv0=1.0,
                 ppv1=1.0):
        """
        Args:
            s_as_input: should the sensitive attribute be part of the input?
            average_prediction: whether to use to average of all possible sensitive attributes for
                                predictions
            npv0: negative predictive value for s=0
            npv1: negative predictive value for s=1
            ppv0: positive predictive value for s=0
            ppv1: positive predictive value for s=1
        """
        super().__init__(s_as_input=s_as_input)
        if s_as_input and average_prediction:
            self.name = f"{self.basename}_cal_av_True"
        else:
            self.name = f"{self.basename}_cal_in_{s_as_input}"
        for val, name in [(npv0, '0npv'), (npv1, '1npv'),
                          (ppv0, '0ppv'), (ppv1, '1ppv')]:
            self.name += f"_{name}_{val}"  # add to name
        self.average_prediction = average_prediction
        self.npv0 = npv0
        self.npv1 = npv1
        self.ppv0 = ppv0
        self.ppv1 = ppv1

    def _additional_parameters(self, raw_data):
        return dict(
            lik='CalibrationLikelihood',
            average_prediction=self.average_prediction,
            p_yybar0_s0=self.npv0,
            p_yybar0_s1=self.npv1,
            p_yybar1_s0=self.ppv0,
            p_yybar1_s1=self.ppv1,
        )


def prepare_data(train_df, test_df, class_attr, positive_class_val, sensitive_attrs,
                 single_sensitive, privileged_vals, params):
    # Separate data
    sensitive = [df[single_sensitive].values[:, np.newaxis] for df in [train_df, test_df]]
    label = [df[class_attr].values[:, np.newaxis] for df in [train_df, test_df]]
    nosensitive = [df.drop(columns=sensitive_attrs).drop(columns=class_attr).values
                   for df in [train_df, test_df]]

    # Check sensitive attributes
    assert list(np.unique(sensitive[0])) == [0, 1] or list(np.unique(sensitive[0])) == [0., 1.]

    # Check labels
    label, label_converter = _fix_labels(label, positive_class_val)
    return dict(xtrain=nosensitive[0], xtest=nosensitive[1], ytrain=label[0], ytest=label[1],
                strain=sensitive[0], stest=sensitive[1]), label_converter, params.get('gpu', 0)


def prior_s(sensitive):
    """Compute the bias in the labels with respect to the sensitive attributes"""
    return np.sum(sensitive == 0) / len(sensitive), np.sum(sensitive == 1) / len(sensitive)


def compute_bias(labels, sensitive):
    """Compute the bias in the labels with respect to the sensitive attributes"""
    rate_y1_s0 = np.sum(labels[sensitive == 0] == 1) / np.sum(sensitive == 0)
    rate_y1_s1 = np.sum(labels[sensitive == 1] == 1) / np.sum(sensitive == 1)
    return rate_y1_s0, rate_y1_s1


def compute_odds(labels, predictions, sensitive):
    """Compute the bias in the predictions with respect to the sensitive attr. and the labels"""
    return dict(
        p_ybary0_s0=np.mean(predictions[np.logical_and(labels == 0, sensitive == 0)] == 0),
        p_ybary1_s0=np.mean(predictions[np.logical_and(labels == 1, sensitive == 0)] == 1),
        p_ybary0_s1=np.mean(predictions[np.logical_and(labels == 0, sensitive == 1)] == 0),
        p_ybary1_s1=np.mean(predictions[np.logical_and(labels == 1, sensitive == 1)] == 1),
    )


def _fix_labels(labels, positive_class_val):
    """Make sure that labels are either 0 or 1

    Args"
        labels: the labels as a list of numpy arrays
        positive_class_val: the value that corresponds to a "positive" predictions

    Returns:
        the fixed labels and a function to convert the fixed labels back to the original format
    """
    label_values = list(np.unique(labels[0]))
    if label_values == [0, 1] and positive_class_val == 1:

        def _do_nothing(inp):
            return inp
        return labels, _do_nothing
    elif label_values == [1, 2] and positive_class_val == 1:

        def _converter(label):
            return 2 - label
        return [2 - y for y in labels], _converter
    raise ValueError("Labels have unknown structure")


def split_train_dev(inputs, labels, sensitive):
    n = inputs.shape[0]
    idx_s0_y0 = np.where((sensitive == 0) & (labels == 0))[0]
    idx_s0_y1 = np.where((sensitive == 0) & (labels == 1))[0]
    idx_s1_y0 = np.where((sensitive == 1) & (labels == 0))[0]
    idx_s1_y1 = np.where((sensitive == 1) & (labels == 1))[0]

    train_fraction = []
    test_fraction = []
    for a in [idx_s0_y0, idx_s0_y1, idx_s1_y0, idx_s1_y1]:
        np.random.shuffle(a)

        split_idx = int(len(a) * 0.5) + 1  # make sure the train part is at least half
        train_fraction_a = a[:split_idx]
        test_fraction_a = a[split_idx:]
        train_fraction += list(train_fraction_a)
        test_fraction += list(test_fraction_a)
    xtrain, ytrain, strain = (inputs[train_fraction], labels[train_fraction],
                              sensitive[train_fraction])
    # ensure that the train set has exactly the same size as the given set
    # (otherwise inducing inputs has wrong shape)
    return dict(xtrain=np.concatenate((xtrain, xtrain))[:n],
                ytrain=np.concatenate((ytrain, ytrain))[:n],
                strain=np.concatenate((strain, strain))[:n], xtest=inputs[test_fraction],
                ytest=labels[test_fraction], stest=sensitive[test_fraction])


def _flags(parameters, data_path, save_dir, s_as_input, model_name, num_train, gpu):
    batch_size = min(MAX_BATCH_SIZE, num_train)
    return {**dict(
        inf='Variational',
        data='sensitive_from_numpy',
        dataset_path=data_path,
        cov='RBFKernel',
        mean='ZeroMean',
        optimizer="Adam",
        lr=0.05,
        # lr=0.1,
        model_name=model_name,
        batch_size=batch_size,
        # epochs=min(MAX_EPOCHS, _num_epochs(num_train)),
        epochs=70,
        eval_epochs=5,
        summary_steps=100000,
        chkpt_epochs=100000,
        save_dir=save_dir,  # "/home/ubuntu/out2/",
        plot='',
        logging_steps=1,
        gpus=str(gpu),
        preds_path='predictions.npz',  # save the predictions into `predictions.npz`
        num_samples=1000,
        optimize_inducing=True,
        length_scale=1.2,
        sf=1.0,
        iso=False,
        num_samples_pred=2000,
        s_as_input=s_as_input,
        # num_inducing=MAX_NUM_INDUCING,
        num_inducing=_num_inducing(num_train),
        manual_seed=SEED,
        metrics=("binary_accuracy,pred_rate_y1_s0,pred_rate_y1_s1,base_rate_y1_s0,base_rate_y1_s1,"
                 "pred_odds_yhaty1_s0,pred_odds_yhaty1_s1,pred_odds_yhaty0_s0,pred_odds_yhaty0_s1")
    ), **parameters}


def _num_inducing(num_train):
    """Adaptive number of inducing inputs

    num_train == 4,000 => num_inducing == 1121
    num_train == 20,000 => num_inducing == 2507
    """
    return int(2500 / 141 * np.sqrt(num_train))


def _num_epochs(num_train):
    """Adaptive number of epochs

    num_train == 4,000 => num_epochs == 125.7
    num_train == 20,000 => num_epochs == 84
    """
    return int(1000 / np.power(num_train, 1 / 4))
