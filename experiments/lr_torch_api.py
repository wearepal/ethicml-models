"""
Logistic Regression in PyTorch
"""

from typing import Optional, Union, List, Dict, Any
from pathlib import Path

import pandas as pd

from ethicml.algorithms.inprocess import InAlgorithmAsync
from ethicml.metrics import TPR, Metric
from ethicml.utility import DataTuple, TestTuple, PathTuple, TestPathTuple
from ethicml.evaluators.per_sensitive_attribute import (
    metric_per_sensitive_attribute,
    ratio_per_sensitive_attribute,
)

from lr_torch_impl import DPFlags, EOFlags, run as lr_torch_run, LrSettings


class LrTorch(InAlgorithmAsync):
    def __init__(
        self,
        weight_decay=1e-1,
        batch_size=64,
        lr_decay=1.0,
        learning_rate=1e-3,
        epochs=100,
        fair=False,
        debiasing_args: Optional[Union[DPFlags, EOFlags]] = None,
        use_sgd=False,
        use_s=False,
        use_gpu=False,
    ):
        super().__init__()
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.fair = fair
        if fair:
            assert debiasing_args is not None
            self.debiasing_args: Union[DPFlags, EOFlags] = debiasing_args
        self.use_sgd = use_sgd
        self.use_s = use_s
        self.use_gpu = use_gpu

    @property
    def name(self):
        name_ = f"LR (torch), wd: {self.weight_decay}"
        if self.use_sgd:
            name_ += ", SGD"
        else:
            name_ += ", RAdam"
        if self.use_s:
            name_ += ", use s"
        if self.fair:
            if isinstance(self.debiasing_args, DPFlags):
                name_ += f", PR_t: {self.debiasing_args.target_rate_s0}"
            else:
                name_ += f", TPR_t: {self.debiasing_args.p_ybary1_s0}"
                name_ += f", TNR_t: {self.debiasing_args.p_ybary0_s0}"
        return name_

    def run(self, train: DataTuple, test: TestTuple) -> pd.DataFrame:
        import torch
        settings = LrSettings(
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            lr_decay=self.lr_decay,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            debiasing_args=self.debiasing_args,
            use_sgd=self.use_sgd,
            use_s=self.use_s,
            device=torch.device("cuda") if self.use_gpu else torch.device("cpu"),
        )
        return lr_torch_run(settings, train, test)

    def _script_command(
        self,
        train_paths: PathTuple,
        test_paths: TestPathTuple,
        pred_path: Path,
    ) -> List[str]:
        debiasing_flags: Dict[str, float] = {}
        if self.fair:
            if isinstance(self.debiasing_args, DPFlags):
                fairness = "DP"
                debiasing_flags['target_rate_s0'] = self.debiasing_args.target_rate_s0
                debiasing_flags['target_rate_s1'] = self.debiasing_args.target_rate_s1
            else:
                fairness = "EO"
                debiasing_flags['p_ybary1_s0'] = self.debiasing_args.p_ybary1_s0
                debiasing_flags['p_ybary1_s1'] = self.debiasing_args.p_ybary1_s1
                debiasing_flags['p_ybary0_s0'] = self.debiasing_args.p_ybary0_s0
                debiasing_flags['p_ybary0_s1'] = self.debiasing_args.p_ybary0_s1
        else:
            fairness = "None"

        flags: Dict[str, Any] = dict(
            train_x=train_paths.x,
            train_s=train_paths.x,
            train_y=train_paths.y,
            train_name=train_paths.name,
            test_x=test_paths.x,
            test_s=test_paths.s,
            test_name=test_paths.name,
            pred_path=pred_path,

            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            lr_decay=self.lr_decay,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            fairness=fairness,
            use_sgd=self.use_sgd,
            use_s=self.use_s,
            use_gpu=self.use_gpu,
        )
        flags.update(debiasing_flags)

        flags_list: List[str] = []

        for key, values in flags.items():
            flags_list.append(f"--{key}")
            flags_list.append(str(values))

        return ["lr_torch_impl.py"] + flags_list


class TPRRatio(Metric):
    """TPR-ratio"""

    def score(self, prediction: pd.DataFrame, actual) -> float:
        per_sens = metric_per_sensitive_attribute(prediction, actual, TPR())
        ratios = ratio_per_sensitive_attribute(per_sens)

        return list(ratios.values())[0]

    @property
    def name(self) -> str:
        return "TPR-ratio"

    @property
    def apply_per_sensitive(self) -> bool:
        return False
