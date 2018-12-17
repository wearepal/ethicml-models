import gpytorch
from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

class GPClassificationModel(AbstractVariationalGP):
    def __init__(self, inducing_inputs, args):
        variational_distribution = CholeskyVariationalDistribution(inducing_inputs.shape[0])
        variational_strategy = VariationalStrategy(self, inducing_inputs, variational_distribution)
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(getattr(gpytorch.kernels, args.cov)())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred
