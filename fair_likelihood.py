import numpy as np
import torch

from gpytorch import settings
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.functions import log_normal_cdf


class TunePrLikelihood(BernoulliLikelihood):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def variational_log_probability(self, latent_func, labels):
        """
        `target` is expected to be two-dimensional: y and s
        y is either -1 or 1
        s is either 0 or 1
        """
        # get samples
        num_samples = settings.num_likelihood_samples.value()
        latent_samples = latent_func.rsample(torch.Size([num_samples])).view(-1)
        # get target and sensitive attribute
        target, sens_attr = torch.unbind(labels, dim=-1)
        target = target.unsqueeze(0).repeat(num_samples, 1).view(-1)
        if self.training:
            sens_attr = sens_attr.unsqueeze(0).repeat(num_samples, 1).view(-1)
            sens_attr = sens_attr.to(torch.int64)
            # convert target to binary values (0 and 1)
            labels_bin = (0.5 * (target + 1)).to(torch.int64)
            log_lik_neg = log_normal_cdf(-latent_samples)
            log_lik_pos = log_normal_cdf(latent_samples)
            log_lik = torch.stack((log_lik_neg, log_lik_pos), dim=-1)
            debias = self._debiasing_parameters()
            if self.args.use_cuda:
                debias = debias.cuda()
            # `debias` has the shape (y * s, y'). we compute the index as (y_index) * 2 + (s_index)
            # then we use this as index for `debias`
            # shape of debias_per_example: (batch_size, 2)
            debias_per_example = torch.index_select(debias, dim=0, index=labels_bin * 2 + sens_attr)
            weighted_lik = debias_per_example * torch.exp(log_lik)
            log_cond_prob = weighted_lik.sum(dim=-1).log()
        else:
            log_cond_prob = log_normal_cdf(latent_samples.mul(target))
        return log_cond_prob.sum().div(num_samples)

    def _debiasing_parameters(self):
        return debiasing_params_target_rate(self.args)


class BaselineLikelihood(BernoulliLikelihood):
    """This is just the BernoulliLikelihood but it ignores the sensitive attributes in the labels"""
    def __init__(self, _):
        super().__init__()

    def variational_log_probability(self, latent_func, labels):
        target, _ = torch.unbind(labels, dim=-1)
        return super().variational_log_probability(latent_func, target)


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
    return torch.from_numpy(label_posterior.astype(np.float32))


def debiasing_params_target_rate(args):
    """Debiasing parameters for implementing target acceptance rates
    Args:
        args: dictionary with parameters
    Returns:
        P(y|y',s) with shape (y, s, y')
    """
    if args.probs_from_flipped:
        biased_acceptance1 = 0.5 * (1 - args.reject_flip_probability)
        biased_acceptance2 = 0.5 * (1 + args.accept_flip_probability)
    else:
        biased_acceptance1 = args.biased_acceptance1
        biased_acceptance2 = args.biased_acceptance2
    # P(y'=1|s)
    target_acceptance = np.array([args.target_rate1, args.target_rate2])
    # P(y=1|s)
    positive_prior = np.array([biased_acceptance1, biased_acceptance2])
    # P(y'=1|y,s) shape: (y, s)
    positive_value = positive_label_likelihood(args, positive_prior, target_acceptance)
    # P(y'|s) shape: (s, y')
    label_evidence = np.stack([1 - target_acceptance, target_acceptance], axis=-1)
    return compute_label_posterior(positive_value, positive_prior, label_evidence)


def positive_label_likelihood(args, biased_acceptance, target_acceptance):
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
        p_ybary1 = args.p_ybary0_or_ybary1_s0 if s == 0 else args.p_ybary0_or_ybary1_s1
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


def debiasing_params_target_tpr(args):
    """Debiasing parameters for targeting TPRs and TNRs
    Args:
        args: dictionary with parameters
    Returns:
        P(y|y',s) with shape (y, s, y')
    """
    # P(y=1|s)
    positive_prior = np.array([args.biased_acceptance1, args.biased_acceptance2])
    # P(y'=1|y=1,s)
    positive_predictive_value = np.array([args.p_ybary1_s0, args.p_ybary1_s1])
    # P(y'=0|y=0,s)
    negative_predictive_value = np.array([args.p_ybary0_s0, args.p_ybary0_s1])
    # P(y'=1|y=0,s)
    false_omission_rate = 1 - negative_predictive_value
    # P(y'=1|y,s) shape: (y, s)
    positive_value = np.stack([false_omission_rate, positive_predictive_value], axis=0)
    return compute_label_posterior(positive_value, positive_prior)
