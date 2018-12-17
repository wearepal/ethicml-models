import torch
import numpy as np
# from matplotlib import pyplot as plt

from gpytorch.mlls.variational_elbo import VariationalELBO

from fair_likelihood import TunePrLikelihood
from flags import parse_arguments
from gp_model import GPClassificationModel
from dataset import from_numpy


def main(args):
    # Toy data:
    # train_x = torch.linspace(0, 1, 10)
    # train_y = torch.sign(torch.cos(train_x * (4 * np.pi)))
    # train_s = 0.5 * (torch.sign(torch.linspace(-1, 1, 10)) + 1)
    # train_labels = torch.stack((train_y, train_s), dim=-1)

    train_ds, test_ds, inducing_inputs = from_numpy(args)
    train_ds = torch.utils.data.DataLoader(train_ds, batch_size=len(train_ds))
    # Initialize model and likelihood
    model = GPClassificationModel(inducing_inputs, args)
    likelihood = TunePrLikelihood(args)
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # "Loss" for GPs - the marginal log likelihood
    # num_data refers to the amount of training data
    mll = VariationalELBO(likelihood, model, len(train_ds))

    for i in range(args.train_steps):
        for sample in train_ds:
            inputs, labels = sample
        print(f"Size of input: {inputs.size(0)}")
        # Zero backpropped gradients from previous iteration
        optimizer.zero_grad()
        # Get predictive output
        output = model(inputs)
        # Calc loss and backprop gradients
        loss = -mll(output, labels)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, args.train_steps, loss.item()))
        optimizer.step()

    # Go into eval mode
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        # Test x are regularly spaced by 0.01 0,1 inclusive
        # test_x = torch.linspace(0, 1, 101)
        # test_y = torch.sign(torch.cos(test_x * (4 * np.pi)))
        correct_preds = 0
        for sample in torch.utils.data.DataLoader(test_ds, batch_size=10):
            inputs, labels = sample
            # Get classification predictions
            observed_pred = likelihood(model(inputs))
            # Get the predicted labels (probabilites of belonging to the positive class)
            # Transform these probabilities to be 0/1 labels
            pred_labels = observed_pred.mean.ge(0.5).float().mul(2).sub(1)
            correct_preds += (labels[:, 0].numpy() == pred_labels.numpy()).astype(np.float32).sum()
        accuracy = correct_preds / len(test_ds)
        print(accuracy)

        # # Initialize fig and axes for plot
        # fig, plot = plt.subplots(figsize=(4, 3))
        # plot.plot(train_x.numpy(), train_y.numpy(), 'k*')
        # plot.plot(test_x.numpy(), pred_labels.numpy(), 'b')
        # plot.set_ylim([-3, 3])
        # plot.legend(['Observed Data', 'Mean', 'Confidence'])


if __name__ == "__main__":
    main(parse_arguments())
