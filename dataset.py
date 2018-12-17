"""Dataset with sensitive attribute from numpy files"""
from pathlib import Path

import torch
from torch.utils.data import TensorDataset
import numpy as np


def from_numpy(args):
    """Load all data from `dataset_path` and then construct a dataset

    You must specify a path to a numpy file in the flag `dataset_path`. This file must contain
    the following numpy arrays: 'xtrain', 'ytrain', 'strain', 'xtest', 'ytest', 'stest'.
    """
    # Load data from `dataset_path`
    raw_data = np.load(Path(args.dataset_path))

    # Normalize input and create DATA tuples for easier handling
    input_normalizer = _get_normalizer(raw_data['xtrain'], args.dataset_standardize)
    train_x = torch.tensor(input_normalizer(raw_data['xtrain']), dtype=torch.float32)
    train_y = torch.tensor(raw_data['ytrain'], dtype=torch.float32)
    train_s = torch.tensor(raw_data['strain'], dtype=torch.float32)
    test_x = torch.tensor(input_normalizer(raw_data['xtest']), dtype=torch.float32)
    test_y = torch.tensor(raw_data['ytest'], dtype=torch.float32)
    test_s = torch.tensor(raw_data['stest'], dtype=torch.float32)

    # # Construct the inducing inputs from the separated data
    inducing_inputs = _inducing_inputs(args.num_inducing, train_x, train_s, args.s_as_input)

    if args.s_as_input:
        train_x = torch.cat((train_x, train_s), dim=1)
        test_x = torch.cat((test_x, test_s), dim=1)
    train_ds = TensorDataset(train_x, torch.cat((train_y, train_s), dim=1))
    test_ds = TensorDataset(test_x, torch.cat((test_y, test_s), dim=1))
    return train_ds, test_ds, inducing_inputs


def _inducing_inputs(max_num_inducing, train_x, train_s, s_as_input):
    """Construct inducing inputs

    This could be done more cleverly with k means

    Args:
        train: the training data
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
        max_per_feature = np.amax(base, axis=0)

        def _normalizer(unnormalized):
            return np.where(max_per_feature > 1e-7, unnormalized / max_per_feature, unnormalized)
        return _normalizer

    def _do_nothing(inp):
        return inp
    return _do_nothing
