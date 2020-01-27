"""Additional train-test splitters."""
from typing import Tuple, Dict

from ethicml.common import implements
from ethicml.preprocessing import RandomSplit, train_test_split, DataSplitter
from ethicml.utility import DataTuple

__all__ = ["TrainTrainSplit"]


class TrainTrainSplit(RandomSplit):
    """Splitter that returns two times the training set."""

    @implements(DataSplitter)
    def __call__(
        self, data: DataTuple, split_id: int = 0
    ) -> Tuple[DataTuple, DataTuple, Dict[str, float]]:
        random_seed = self._get_seed(split_id)
        split_info: Dict[str, float] = {'seed': random_seed}
        train, _ = train_test_split(data, self.train_percentage, random_seed)
        return train, train, split_info
