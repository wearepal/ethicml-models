"""
Logistic Regression in PyTorch
"""

from typing import NamedTuple, Optional, Union
import math
import random

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import SGD
from torch.optim.optimizer import Optimizer, required

import numpy as np
import pandas as pd

from ethicml.algorithms.inprocess import InAlgorithm
from ethicml.implementations.pytorch_common import CustomDataset, TestDataset
from ethicml.metrics import TPR, Metric
from ethicml.utility import DataTuple, TestTuple
from ethicml.evaluators.per_sensitive_attribute import (
    metric_per_sensitive_attribute,
    ratio_per_sensitive_attribute,
)


class RAdam(Optimizer):
    """Rectified Adam optimizer"""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state["step"] += 1
                buffered = self.buffer[int(state["step"] % 10)]
                if state["step"] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = (
                            group["lr"]
                            * math.sqrt(
                                (1 - beta2_t)
                                * (N_sma - 4)
                                / (N_sma_max - 4)
                                * (N_sma - 2)
                                / N_sma
                                * N_sma_max
                                / (N_sma_max - 2)
                            )
                            / (1 - beta1 ** state["step"])
                        )
                    else:
                        step_size = group["lr"] / (1 - beta1 ** state["step"])
                    buffered[2] = step_size

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(-group["weight_decay"] * group["lr"], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


class EOFlags(NamedTuple):
    p_ybary1_s0: float
    p_ybary1_s1: float
    p_ybary0_s0: float
    p_ybary0_s1: float
    biased_acceptance_s0: Optional[float] = None
    biased_acceptance_s1: Optional[float] = None


class DPFlags(NamedTuple):
    target_rate_s0: float
    target_rate_s1: float
    p_ybary0_or_ybary1_s0: float = 1.0
    p_ybary0_or_ybary1_s1: float = 1.0
    biased_acceptance_s0: Optional[float] = None
    biased_acceptance_s1: Optional[float] = None


def compute_label_posterior(positive_value, positive_prior, label_evidence=None):
    """Return label posterior from positive likelihood P(y'=1|y,s) and positive prior P(y=1|s)
    Args:
        positive_value: P(y'=1|y,s), shape (y, s)
        label_prior: P(y|s)
    Returns:
        Label posterior, shape (y, s, y')
    """
    # compute the prior
    # P(y=0|s)
    negative_prior = 1 - positive_prior
    # P(y|s) shape: (y, s, 1)
    label_prior = np.stack([negative_prior, positive_prior], axis=0)[..., np.newaxis]

    # compute the likelihood
    # P(y'|y,s) shape: (y, s, y')
    label_likelihood = np.stack([1 - positive_value, positive_value], axis=-1)

    # compute joint and evidence
    # P(y',y|s) shape: (y, s, y')
    joint = label_likelihood * label_prior
    # P(y'|s) shape: (s, y')
    if label_evidence is None:
        label_evidence = np.sum(joint, axis=0)

    # compute posterior
    # P(y|y',s) shape: (y, s, y')
    label_posterior = joint / label_evidence
    # reshape to (y * s, y') so that we can use gather on the first dimension
    label_posterior = np.reshape(label_posterior, (4, 2))
    # take logarithm because we need that anyway later
    log_label_posterior = np.log(label_posterior)
    return torch.from_numpy(log_label_posterior.astype(np.float32))


def debiasing_params_target_tpr(flags: EOFlags):
    """Debiasing parameters for targeting TPRs and TNRs
    Args:
        flags: object with parameters
    Returns:
        P(y|y',s) with shape (y, s, y')
    """
    # P(y=1|s)
    positive_prior = np.array([flags.biased_acceptance_s0, flags.biased_acceptance_s1])
    # P(y'=1|y=1,s)
    positive_predictive_value = np.array([flags.p_ybary1_s0, flags.p_ybary1_s1])
    # P(y'=0|y=0,s)
    negative_predictive_value = np.array([flags.p_ybary0_s0, flags.p_ybary0_s1])
    # P(y'=1|y=0,s)
    false_omission_rate = 1 - negative_predictive_value
    # P(y'=1|y,s) shape: (y, s)
    positive_value = np.stack([false_omission_rate, positive_predictive_value], axis=0)
    return compute_label_posterior(positive_value, positive_prior)


def debiasing_params_target_rate(flags: DPFlags):
    """Debiasing parameters for implementing target acceptance rates
    Args:
        flags: object with parameters
    Returns:
        P(y|y',s) with shape (y, s, y')
    """
    biased_acceptance_s0 = flags.biased_acceptance_s0
    biased_acceptance_s1 = flags.biased_acceptance_s1
    # P(y'=1|s)
    target_acceptance = np.array([flags.target_rate_s0, flags.target_rate_s1])
    # P(y=1|s)
    positive_prior = np.array([biased_acceptance_s0, biased_acceptance_s1])
    # P(y'=1|y,s) shape: (y, s)
    positive_value = positive_label_likelihood(flags, positive_prior, target_acceptance)
    # P(y'|s) shape: (s, y')
    label_evidence = np.stack([1 - target_acceptance, target_acceptance], axis=-1)
    return compute_label_posterior(positive_value, positive_prior, label_evidence)


def positive_label_likelihood(flags: DPFlags, biased_acceptance, target_acceptance):
    """Compute the label likelihood (for positive labels)
    Args:
        biased_acceptance: P(y=1|s)
        target_acceptance: P(y'=1|s)
    Returns:
        P(y'=1|y,s) with shape (y, s)
    """
    positive_lik = []
    for s, (target, biased) in enumerate(zip(target_acceptance, biased_acceptance)):
        # P(y'=1|y=1)
        p_ybary1 = flags.p_ybary0_or_ybary1_s0 if s == 0 else flags.p_ybary0_or_ybary1_s1
        if target > biased:
            # P(y'=1|y=0) = (P(y'=1) - P(y'=1|y=1)P(y=1))/P(y=0)
            p_ybar1_y0 = (target - p_ybary1 * biased) / (1 - biased)
        else:
            p_ybar1_y0 = 1 - p_ybary1
            # P(y'=1|y=0) = (P(y'=1) - P(y'=1|y=0)P(y=0))/P(y=1)
            p_ybary1 = (target - p_ybar1_y0 * (1 - biased)) / biased
        positive_lik.append([p_ybar1_y0, p_ybary1])
    positive_lik_arr = np.array(positive_lik)  # shape: (s, y)
    return np.transpose(positive_lik_arr)  # shape: (y, s)


def fair_loss(logits, sens_attr, target, log_debias):
    sens_attr = sens_attr.to(torch.int64)
    labels_bin = target.to(torch.int64)

    log_lik_neg = F.logsigmoid(-logits)
    log_lik_pos = F.logsigmoid(logits)
    # `log_lik` has shape [num_samples, batch_size, 2]
    log_lik = torch.stack((log_lik_neg, log_lik_pos), dim=-1)

    # `log_debias` has shape [y * s, y']
    # we compute the index as (y_index) * 2 + (s_index)
    # then we use this as index for `log_debias`
    # shape of log_debias_per_example: [batch_size, 2]
    log_debias_per_example = torch.index_select(
        input=log_debias, dim=0, index=labels_bin * 2 + sens_attr
    )

    weighted_log_lik = log_debias_per_example + log_lik
    return -weighted_log_lik.logsumexp(dim=-1)


class LrTorch(InAlgorithm):
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
        self.device = torch.device("cpu")
        self.use_s = use_s

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

    def run(self, train: DataTuple, test: TestTuple):
        seed = 42
        np.random.seed(seed)  # cpu vars
        torch.manual_seed(seed)  # cpu  vars
        random.seed(seed)  # Python

        in_dim = train.x.shape[1]
        if self.use_s:
            train = train.make_copy_with(x=pd.concat([train.x, train.s], axis="columns"))
            test = test.make_copy_with(x=pd.concat([test.x, test.s], axis="columns"))
            in_dim += 1
        train_ds = CustomDataset(train)
        test_ds = TestDataset(test)
        train_ds = DataLoader(train_ds, batch_size=self.batch_size, pin_memory=True, shuffle=True)
        test_ds = DataLoader(test_ds, batch_size=10000, pin_memory=True)

        if self.fair:
            debiasing_args = self.debiasing_args
            if debiasing_args.biased_acceptance_s0 is None:
                biased_acceptance_s0 = float(
                    train.y[train.y.columns[0]].loc[train.s[train.s.columns[0]] == 0].mean()
                )
                debiasing_args = debiasing_args._replace(biased_acceptance_s0=biased_acceptance_s0)
            if debiasing_args.biased_acceptance_s1 is None:
                biased_acceptance_s1 = float(
                    train.y[train.y.columns[0]].loc[train.s[train.s.columns[0]] == 1].mean()
                )
                debiasing_args = debiasing_args._replace(biased_acceptance_s1=biased_acceptance_s1)
            # print(debiasing_args)
            if isinstance(debiasing_args, DPFlags):
                self.debiasing_params = debiasing_params_target_rate(debiasing_args)
            else:
                self.debiasing_params = debiasing_params_target_tpr(debiasing_args)

        model = nn.Linear(in_dim, 1)
        model.to(self.device)
        optimizer: Optimizer
        if self.use_sgd:
            optimizer = SGD(
                model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        else:
            optimizer = RAdam(
                model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        self._fit(
            model=model,
            train_data=train_ds,
            optimizer=optimizer,
            # lr_milestones=dict(milestones=[30, 60, 90, 120], gamma=0.3),
        )
        predictions = self.predict_dataset(model, test_ds)
        return pd.DataFrame(predictions.numpy(), columns=["preds"])

    def _fit(self, model, train_data, optimizer, lr_milestones: Optional[dict] = None):
        scheduler = None
        if lr_milestones is not None:
            scheduler = MultiStepLR(optimizer=optimizer, **lr_milestones)

        for epoch in range(self.epochs):
            # print(f"===> Epoch {epoch} of classifier training")

            for x, s, y in train_data:
                target = y
                x = x.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                logits = model(x)

                if self.fair:
                    logits = logits.view(-1)
                    s = s.to(self.device)
                    s = s.view(-1)
                    target = target.view(-1)
                    losses = fair_loss(logits, s, target, self.debiasing_params)
                else:
                    logits = logits.view(-1, 1)
                    targets = target.view(-1, 1)
                    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
                loss = losses.sum() / x.size(0)

                loss.backward()
                optimizer.step()

            if scheduler is not None:
                scheduler.step(epoch)

    def predict_dataset(self, model, data):
        preds = []
        with torch.set_grad_enabled(False):
            for x, s in data:
                x = x.to(self.device)

                outputs = model(x)
                batch_preds = torch.round(outputs.sigmoid())
                preds.append(batch_preds)

        preds = torch.cat(preds, dim=0).cpu().detach().view(-1)
        return preds


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
