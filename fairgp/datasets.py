"""Dataset with sensitive attribute from numpy files"""
from pathlib import Path

import torch
from torch.utils.data import TensorDataset
import numpy as np


def sensitive_from_numpy(flags):
    """Load all data from `dataset_path` and then construct a dataset

    You must specify a path to a numpy file in the flag `dataset_path`. This file must contain
    the following numpy arrays: 'xtrain', 'ytrain', 'strain', 'xtest', 'ytest', 'stest'.
    """
    # Load data from `dataset_path`
    raw_data = np.load(Path(flags.dataset_path))

    # Normalize input and create DATA tuples for easier handling
    input_normalizer = _get_normalizer(raw_data['xtrain'], flags.dataset_standardize)
    train_x = torch.tensor(input_normalizer(raw_data['xtrain']), dtype=torch.float32)
    train_y = torch.tensor(raw_data['ytrain'], dtype=torch.float32)
    train_s = torch.tensor(raw_data['strain'], dtype=torch.float32)
    test_x = torch.tensor(input_normalizer(raw_data['xtest']), dtype=torch.float32)
    test_y = torch.tensor(raw_data['ytest'], dtype=torch.float32)
    test_s = torch.tensor(raw_data['stest'], dtype=torch.float32)

    train_y = _fix_labels(train_y)
    test_y = _fix_labels(test_y)

    # # Construct the inducing inputs from the separated data
    inducing_inputs = _inducing_inputs(
        flags.num_inducing, train_x, train_s, flags.s_as_input).clone()

    if flags.s_as_input:
        train_x = torch.cat((train_x, train_s), dim=1)
        test_x = torch.cat((test_x, test_s), dim=1)
    train_ds = TensorDataset(train_x, torch.cat((train_y, train_s), dim=1))
    test_ds = TensorDataset(test_x, torch.cat((test_y, test_s), dim=1))
    return train_ds, test_ds, inducing_inputs


def _inducing_inputs(max_num_inducing, train_x, train_s, s_as_input):
    """Construct inducing inputs

    This could be done more cleverly with k means

    Args:
        max_num_inducing: the desired maximum number of inducing inputs
        train_x: the training inputs
        train_s: the training sensitive attributes
        s_as_input: whether or not the sensitive attribute is part of the input

    Returns:
        inducing inputs
    """
    num_train = train_x.size(0)
    num_inducing = min(num_train, max_num_inducing)
    if s_as_input:
        return torch.cat((train_x[::num_train // num_inducing],
                          train_s[::num_train // num_inducing]), -1)
    return train_x[::num_train // num_inducing]


def _get_normalizer(base, do_standardize):
    """Construct normalizer to prevent Cholesky problems"""
    if do_standardize:
        mean, std = np.mean(base, axis=0), np.std(base, axis=0)
        std[std < 1e-7] = 1.

        def _standardizer(unstandardized):
            return (unstandardized - mean) / std
        return _standardizer
    elif base.min() == 0 and base.max() > 10:
        print("Doing normalization...")
        max_per_feature = np.amax(base, axis=0)

        def _normalizer(unnormalized):
            return np.where(max_per_feature > 1e-7, unnormalized / max_per_feature, unnormalized)
        return _normalizer

    def _do_nothing(inp):
        return inp
    return _do_nothing


def _fix_labels(labels):
    if labels.min() == 0 and labels.max() == 1:
        print("Fixing labels...")
        return 2 * labels - 1
    return labels


def toy_data_1d_multitask(flags):
    """Example with multi-dimensional output."""
    n_all = 200
    num_train = 50
    num_inducing = min(num_train, flags.num_inducing)

    inputs = np.linspace(0, 5, num=n_all)[:, np.newaxis]
    output1 = np.cos(inputs)
    output2 = np.sin(inputs)
    outputs = np.concatenate((output1, output2), axis=1)

    np.random.seed(4)
    (xtrain, ytrain), (xtest, ytest) = select_training_and_test(num_train, inputs, outputs)

    xtrain = torch.tensor(xtrain, dtype=torch.float32)
    ytrain = torch.tensor(ytrain, dtype=torch.float32)
    xtest = torch.tensor(xtest, dtype=torch.float32)
    ytest = torch.tensor(ytest, dtype=torch.float32)
    train_ds = TensorDataset(xtrain, ytrain)
    test_ds = TensorDataset(xtest, ytest)
    return train_ds, test_ds, xtrain[::num_train // num_inducing]


def toy_data_1d(flags):
    """Simple 1D example with synthetic data."""
    n_all = 200
    num_train = 50
    num_inducing = flags.num_inducing

    inputs = np.linspace(0, 5, num=n_all)
    outputs = np.cos(inputs)
    inputs = inputs[:, np.newaxis]
    np.random.seed(4)
    (xtrain, ytrain), (xtest, ytest) = select_training_and_test(num_train, inputs, outputs)

    xtrain = torch.tensor(xtrain, dtype=torch.float32)
    ytrain = torch.tensor(ytrain, dtype=torch.float32)
    xtest = torch.tensor(xtest, dtype=torch.float32)
    ytest = torch.tensor(ytest, dtype=torch.float32)
    train_ds = TensorDataset(xtrain, ytrain)
    test_ds = TensorDataset(xtest, ytest)
    return train_ds, test_ds, xtrain[::num_train // num_inducing]


def select_training_and_test(num_train, *data_parts):
    """Randomly devide a dataset into training and test
    Args:
        num_train: Desired number of examples in training set
        *data_parts: Parts of the dataset. The * means that the function can take an arbitrary
                     number of arguments.
    Returns:
        Two lists: data_parts_train, data_parts_test
    """
    idx = np.arange(data_parts[0].shape[0])
    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = np.sort(idx[num_train:])

    data_parts_train = []
    data_parts_test = []
    for data_part in data_parts:  # data_parts is a list of the arguments passed to the function
        data_parts_train.append(data_part[train_idx])
        data_parts_test.append(data_part[test_idx])

    return data_parts_train, data_parts_test
