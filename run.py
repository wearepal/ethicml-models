import torch
import numpy as np
# from matplotlib import pyplot as plt

import gpytorch
from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.mlls.variational_elbo import VariationalELBO


class GPClassificationModel(AbstractVariationalGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution)
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


def main():
    train_x = torch.linspace(0, 1, 10)
    train_y = torch.sign(torch.cos(train_x * (4 * np.pi)))
    # Initialize model and likelihood
    model = GPClassificationModel(train_x)
    likelihood = gpytorch.likelihoods.BernoulliLikelihood()
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    # num_data refers to the amount of training data
    mll = VariationalELBO(likelihood, model, train_y.numel())

    training_iter = 50
    for i in range(training_iter):
        # Zero backpropped gradients from previous iteration
        optimizer.zero_grad()
        # Get predictive output
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()

    # Go into eval mode
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        # Test x are regularly spaced by 0.01 0,1 inclusive
        test_x = torch.linspace(0, 1, 101)
        test_y = torch.sign(torch.cos(test_x * (4 * np.pi)))
        # Get classification predictions
        observed_pred = likelihood(model(test_x))
        # Get the predicted labels (probabilites of belonging to the positive class)
        # Transform these probabilities to be 0/1 labels
        pred_labels = observed_pred.mean.ge(0.5).float().mul(2).sub(1)
        accuracy = (test_y.numpy() == pred_labels.numpy()).astype(np.float32).mean()
        print(accuracy)

        # # Initialize fig and axes for plot
        # fig, plot = plt.subplots(figsize=(4, 3))
        # plot.plot(train_x.numpy(), train_y.numpy(), 'k*')
        # plot.plot(test_x.numpy(), pred_labels.numpy(), 'b')
        # plot.set_ylim([-3, 3])
        # plot.legend(['Observed Data', 'Mean', 'Confidence'])


if __name__ == "__main__":
    main()
