"""Additional train-test splitters."""
from typing import Tuple, Dict

from kit import implements
from ethicml import ProportionalSplit, DataSplitter
from ethicml import DataTuple

__all__ = ["TrainTrainSplit"]


class TrainTrainSplit(ProportionalSplit):
    """Splitter that returns two times the training set."""

    @implements(DataSplitter)
    def __call__(
        self, data: DataTuple, split_id: int = 0
    ) -> Tuple[DataTuple, DataTuple, Dict[str, float]]:
        train, _, split_info = super().__call__(data, split_id=split_id)
        return train, train, split_info
