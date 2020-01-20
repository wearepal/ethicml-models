"""Definitions of GP models"""
import gpytorch
from gpytorch.models import AbstractVariationalGP, ExactGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
    GridInterpolationVariationalStrategy,
)

from .utils import utils


class Variational(AbstractVariationalGP):
    """GP using variational inference"""

    def __init__(self, inducing_inputs, kernel, mean, flags, **kwargs):
        num_inducing = inducing_inputs.shape[0]
        print(f"num inducing from shape: {num_inducing}")

        if flags.cov == 'GridInterpolationKernel':
            num_dim = inducing_inputs.shape[1]
            grid_bounds = (-10.0, 10.0)
            grid_size = 400
            # define the variational distribution with the grid_size
            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=grid_size, batch_size=num_dim
            )
            # The variational strategy defines how the GP prior is computed and
            # how to marginalize out the function values (only for DKL)
            variational_strategy = GridInterpolationVariationalStrategy(
                self,
                #                 inducing_inputs,
                grid_size=grid_size,
                grid_bounds=[grid_bounds],
                #                 num_dim=num_dim,
                variational_distribution=variational_distribution,
                #                 mixing_params=False, sum_output=False,
                #                 learn_inducing_locations=flags.optimize_inducing
            )
        else:
            # define the variational distribution with the length of the inducing inputs
            variational_distribution = CholeskyVariationalDistribution(num_inducing)

            # The variational strategy defines how the GP prior is computed and
            # how to marginalize out the inducing point function values
            variational_strategy = VariationalStrategy(
                self,
                inducing_inputs,
                variational_distribution,
                learn_inducing_locations=flags.optimize_inducing,
            )

        super().__init__(variational_strategy)
        # initialize mean and covariance
        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = mean
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred

    def get_marginal_log_likelihood(self, likelihood, num_data):
        """Get the marginal log likelihood function that works with the GP model"""
        return gpytorch.mlls.VariationalELBO(likelihood, self, num_data)


class Exact(ExactGP):
    """GP using exact inference"""

    def __init__(self, train_ds, likelihood, kernel, mean, **kwargs):
        train_x, train_y = utils.dataset2tensor(train_ds)
        train_x, train_y = train_x.squeeze(-1), train_y.squeeze(-1)
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred

    def get_marginal_log_likelihood(self, likelihood, _):
        """Get the marginal log likelihood function that works with the GP model"""
        return gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self)


class ExactMultitask(ExactGP):
    """GP using exact inference with multi-dimensional output"""

    def __init__(self, train_ds, likelihood, kernel, mean, **kwargs):
        train_x, train_y = utils.dataset2tensor(train_ds)
        train_x, train_y = train_x.squeeze(-1), train_y.squeeze(-1)
        super().__init__(train_x, train_y, likelihood)
        num_tasks = train_y.size()[1]
        self.mean_module = gpytorch.means.MultitaskMean(mean, num_tasks=num_tasks)
        self.covar_module = gpytorch.kernels.MultitaskKernel(kernel, num_tasks=num_tasks, rank=1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
        return latent_pred

    def get_marginal_log_likelihood(self, likelihood, _):
        """Get the marginal log likelihood function that works with the GP model"""
        return gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self)
