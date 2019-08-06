"""Plotting functions"""
import numpy as np
from matplotlib import pyplot as plt

from .utils import dataset2numpy


def simple_1d(pred_mean, pred_var, train_ds, test_ds, in_dim=0):
    """Plot train and test data and predicted data with uncertainty."""
    xtrain, ytrain = dataset2numpy(train_ds)
    xtest, ytest = dataset2numpy(test_ds)
    flexible_1d(xtest, (pred_mean, pred_var), (xtrain, ytrain), (xtest, ytest), in_dim)


def flexible_1d(xpreds, preds, train, test, in_dim=0):
    """Plot train and test data and predicted data with uncertainty.

    Args:
        xpreds: inputs for predictions
        preds: predictions
        train: training inputs and outputs
        test: testing inputs and outputs
        in_dim: (optional) the input dimension that will be plotted
    """
    xtrain, ytrain = train
    xtest, ytest = test
    pred_mean, pred_var = preds
    sorted_index = np.argsort(xpreds[:, 0])
    xpreds = xpreds[sorted_index]
    pred_mean = pred_mean[sorted_index]
    pred_var = pred_var[sorted_index]
    out_dims = len(ytrain[0])
    fig, plots = plt.subplots(nrows=out_dims, ncols=1, squeeze=False)
    plots = plots[:, 0]
    for i, plot in enumerate(plots):
        plot.plot(xtrain[:, in_dim], ytrain[:, i], '.', mew=2, label='trainings')
        plot.plot(xtest[:, in_dim], ytest[:, i], 'o', mew=2, label='tests')
        plot.plot(xpreds[:, in_dim], pred_mean[:, i], 'x', mew=2, label='predictions')

        upper_bound = pred_mean[:, i] + 1.96 * np.sqrt(pred_var[:, i])
        lower_bound = pred_mean[:, i] - 1.96 * np.sqrt(pred_var[:, i])

        plot.fill_between(
            xpreds[:, in_dim], lower_bound, upper_bound, color='gray', alpha=0.25, label='95% CI'
        )
    fig.legend(loc='lower left')
    fig.show()
    input("Press Enter to continue...")
