"""FairLR tests."""
import pandas as pd
import numpy as np  # pylint: disable=unused-import  # import needed for mypy

from ethicml.algorithms.inprocess import InAlgorithm
from ethicml.utility import TrainTestPair

from ethicml_models import TuningLr


def count_true(mask: "np.ndarray[np.bool_]") -> int:
    """Count the number of elements that are True"""
    return mask.nonzero()[0].shape[0]


def test_tuning_lr(toy_train_test: TrainTestPair):
    """Test an tuning lr"""
    train, test = toy_train_test
    model: InAlgorithm = TuningLr()

    assert isinstance(model, InAlgorithm)
    assert model is not None
    assert model.name == "TuningLR, wd: 0.1, RAdam"

    predictions: pd.DataFrame = model.run(train, test)
    assert count_true(predictions.values == 1) == 198
    assert count_true(predictions.values == 0) == 202
