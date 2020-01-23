from ethicml.utility import DataTuple, Prediction
from ethicml.metrics import Metric, TPR
from ethicml.evaluators import metric_per_sensitive_attribute, ratio_per_sensitive_attribute

__all__ = ["TPRRatio"]


class TPRRatio(Metric):
    """TPR-ratio"""

    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        per_sens = metric_per_sensitive_attribute(prediction, actual, TPR())
        ratios = ratio_per_sensitive_attribute(per_sens)

        return list(ratios.values())[0]

    @property
    def name(self) -> str:
        return "TPR-ratio"

    @property
    def apply_per_sensitive(self) -> bool:
        return False
