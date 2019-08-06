"""Construct needed GPyTorch objects"""
import numpy as np
import torch
import gpytorch

from . import likelihoods, gp_models, datasets


def construct_model(flags):
    """Construct GP object, likelihood and marginal log likelihood function from the flags

    This function just constructs all the needed objects and gives them all the needed parameters.

    Args:
        flags: settings object
    """
    data_func, Likelihood, Kernel, Mean, GPModel, Optimizer = flags_to_python(flags)

    # Get dataset objects
    train_ds, test_ds, inducing_inputs = data_func(flags)
    input_dim = inducing_inputs.shape[1]

    if flags.use_cuda:
        inducing_inputs = inducing_inputs.cuda()

    # Initialize likelihood, kernel and model
    likelihood: gpytorch.likelihoods.Likelihood = Likelihood(flags)

    # Initialize kernel
    lengthscale_transf, inv_lengthscale_transf = get_lengthscale_transforms(flags.length_scale)
    kernel_unscaled: gpytorch.kernels.Kernel = Kernel(
        ard_num_dims=None if flags.iso else input_dim,
        param_transform=lengthscale_transf,
        inv_param_transform=inv_lengthscale_transf,
    )
    kernel = gpytorch.kernels.ScaleKernel(kernel_unscaled)

    # Initialize kernel
    mean = Mean()

    # Initialize model
    model: gpytorch.models.GP = GPModel(
        train_ds=train_ds,
        likelihood=likelihood,
        kernel=kernel,
        inducing_inputs=inducing_inputs,
        mean=mean,
        flags=flags,
    )
    if flags.use_cuda:
        model, likelihood = model.cuda(), likelihood.cuda()

    # "Loss" for the GP model: the marginal log likelihood
    mll = model.get_marginal_log_likelihood(likelihood, len(train_ds))

    # Initialize optimizer
    optimizer: torch.optim.Optimizer = Optimizer(
        list(model.parameters()) + list(likelihood.parameters()), lr=flags.lr
    )
    return model, likelihood, mll, optimizer, train_ds, test_ds


def flags_to_python(flags):
    """Use the built-in function "getattr" to turn the flags into Python references

    Args:
        flags: settings object
    """
    data_func = getattr(datasets, flags.data)
    likelihood_class = getattr(likelihoods, flags.lik)
    kernel_class = getattr(gpytorch.kernels, flags.cov)
    mean_class = getattr(gpytorch.means, flags.mean)
    model_class = getattr(gp_models, flags.inf)
    optimizer_class = getattr(torch.optim, flags.optimizer)
    return data_func, likelihood_class, kernel_class, mean_class, model_class, optimizer_class


def get_lengthscale_transforms(initial_value):
    """
    Return two functions. The first transforms the raw lengthscale to the actual value.
    The second transorms it back.
    """
    offset = torch.tensor(np.log(np.exp(initial_value) - 1), dtype=torch.float32)

    # @torch.jit.script
    def _lengthscale_transform(raw_lengthscale):
        return torch.nn.functional.softplus(raw_lengthscale + offset)

    # @torch.jit.script
    def _inv_lengthscale_transform(lengthscale):
        return lengthscale.exp().sub(1).log().sub(offset)

    return _lengthscale_transform, _inv_lengthscale_transform
