"""Fair Logistic Regression in PyTorch."""

import random
from typing import NamedTuple, Optional, Union
from typing_extensions import Literal

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import SGD

import numpy as np
import pandas as pd

from ethicml.implementations.pytorch_common import CustomDataset, TestDataset
from ethicml.implementations.utils import InAlgoArgs, load_data_from_flags, save_predictions
from ethicml.utility import DataTuple, TestTuple

from optimizer import RAdam

FairnessType = Literal["DP", "EO"]


class TuningLrArgs(InAlgoArgs):
    weight_decay: float
    batch_size: int
    lr_decay: float
    learning_rate: float
    epochs: int
    use_sgd: bool
    use_s: bool
    device: str
    seed: int
    fairness: Optional[FairnessType] = None
    param1: Optional[float] = None
    param2: Optional[float] = None
    param3: Optional[float] = None
    param4: Optional[float] = None
    biased_acceptance_s0: Optional[float] = None
    biased_acceptance_s1: Optional[float] = None

    def process_args(self) -> None:
        super().process_args()
        if self.fairness is not None:
            if any(p is None for p in (self.param1, self.param2, self.param3, self.param4)):
                raise ValueError("if fairness type is specified, the parameters have to be given")


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


def compute_label_posterior(positive_value, positive_prior, label_evidence=None) -> torch.Tensor:
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


def debiasing_params_target_tpr(flags: EOFlags) -> torch.Tensor:
    """Debiasing parameters for targeting TPRs and TNRs
    Args:
        flags: object with parameters
    Returns:
        P(y|y',s) with shape (y, s, y')
    """
    assert flags.biased_acceptance_s0 is not None and flags.biased_acceptance_s1 is not None
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


def debiasing_params_target_rate(flags: DPFlags) -> torch.Tensor:
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


def run(
    args: TuningLrArgs,
    debiasing_args: Union[None, DPFlags, EOFlags],
    train: DataTuple,
    test: TestTuple,
    device,
    use_cuda: bool,
) -> pd.DataFrame:
    np.random.seed(args.seed)  # cpu vars
    torch.manual_seed(args.seed)  # cpu  vars
    random.seed(args.seed)  # Python
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False

    in_dim = train.x.shape[1]
    if args.use_s:
        train = train.replace(x=pd.concat([train.x, train.s], axis="columns"))
        test = test.replace(x=pd.concat([test.x, test.s], axis="columns"))
        in_dim += 1
    train_ds = CustomDataset(train)
    test_ds = TestDataset(test)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=10000, pin_memory=True)

    debiasing_params = None
    if debiasing_args is not None:
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
            debiasing_params = debiasing_params_target_rate(debiasing_args)
        else:
            debiasing_params = debiasing_params_target_tpr(debiasing_args)

    model = nn.Linear(in_dim, 1)
    model.to(device)
    optimizer: Optimizer
    if args.use_sgd:
        optimizer = SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = RAdam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    fit(
        model=model,
        train_data=train_dl,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device,
        debiasing_params=debiasing_params,
        # lr_milestones=dict(milestones=[30, 60, 90, 120], gamma=0.3),
    )
    predictions = predict_dataset(model, test_dl, device)
    return pd.DataFrame(predictions.numpy(), columns=["preds"])


def fit(
    model,
    train_data: DataLoader,
    optimizer,
    epochs: int,
    device,
    debiasing_params: Optional[torch.Tensor] = None,
    lr_milestones: Optional[dict] = None,
):
    scheduler = None
    if lr_milestones is not None:
        scheduler = MultiStepLR(optimizer=optimizer, **lr_milestones)

    for epoch in range(epochs):
        # print(f"===> Epoch {epoch} of classifier training")

        for x, s, y in train_data:
            target = y
            x = x.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            logits = model(x)

            if debiasing_params is not None:
                logits = logits.view(-1)
                s = s.to(device)
                s = s.view(-1)
                target = target.view(-1)
                losses = fair_loss(logits, s, target, debiasing_params)
            else:
                logits = logits.view(-1, 1)
                targets = target.view(-1, 1)
                losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            loss = losses.sum() / x.size(0)

            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step(epoch)


def predict_dataset(model, data, device):
    preds = []
    with torch.set_grad_enabled(False):
        for x, s in data:
            x = x.to(device)

            outputs = model(x)
            batch_preds = torch.round(outputs.sigmoid())
            preds.append(batch_preds)

    preds = torch.cat(preds, dim=0).cpu().detach().view(-1)
    return preds


def main():
    args = TuningLrArgs(explicit_bool=True).parse_args()
    debiasing_args: Union[None, DPFlags, EOFlags] = None
    if args.fairness == "DP":
        debiasing_args = DPFlags(
            target_rate_s0=args.param1,
            target_rate_s1=args.param2,
            p_ybary0_or_ybary1_s0=args.param3,
            p_ybary0_or_ybary1_s1=args.param4,
            biased_acceptance_s0=args.biased_acceptance_s0,
            biased_acceptance_s1=args.biased_acceptance_s1,
        )
    elif args.fairness == "EO":
        debiasing_args = EOFlags(
            p_ybary1_s0=args.param1,
            p_ybary1_s1=args.param2,
            p_ybary0_s0=args.param3,
            p_ybary0_s1=args.param4,
            biased_acceptance_s0=args.biased_acceptance_s0,
            biased_acceptance_s1=args.biased_acceptance_s1,
        )
    train, test = load_data_from_flags(args)
    device = torch.device(args.device)
    preds = run(
        args=args,
        debiasing_args=debiasing_args,
        train=train,
        test=test,
        device=device,
        use_cuda=args.device.lower() == "cpu",
    )
    save_predictions(preds, args)


if __name__ == "__main__":
    main()
