"""Definitions of GP models"""
import gpytorch
from gpytorch.models import AbstractVariationalGP, ExactGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy


class Variational(AbstractVariationalGP):
    """GP using variational inference"""
    def __init__(self, inducing_inputs, args):
        variational_distribution = CholeskyVariationalDistribution(inducing_inputs.shape[0])

        # The variational strategy defines how the GP prior is computed and
        # how to marginalize out the inducing point function values
        variational_strategy = VariationalStrategy(self, inducing_inputs, variational_distribution,
                                                   learn_inducing_locations=args.optimize_inducing)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(getattr(gpytorch.kernels, args.cov)())

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
    def __init__(self, train_x, train_y, likelihood, args):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(getattr(gpytorch.kernels, args.cov)())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred

    def get_marginal_log_likelihood(self, likelihood, _):
        """Get the marginal log likelihood function that works with the GP model"""
        return gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self)
