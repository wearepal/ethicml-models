"""Defines methods for metrics"""

import numpy as np
from torchnet.meter import AverageValueMeter, ClassErrorMeter


def init_metrics(metric_flag):
    """Initialize metrics

    Args:
        metric_flag: a string that contains the names of the metrics separated with commas
        is_eager: True if in eager execution
    Returns:
        a dictionary with the initialized metrics
    """
    metrics = {}
    if metric_flag == "":
        return metrics  # No metric names given -> return empty dictionary

    # First, find all metrics
    dict_of_metrics = {}
    for class_name in dir(util.metrics):  # Loop over everything that is defined in `metrics`
        class_ = getattr(util.metrics, class_name)
        # Here, we filter out all functions and other classes which are not metrics
        if isinstance(class_, type(Metric)) and issubclass(class_, Metric):
            dict_of_metrics[class_.name] = class_

    if isinstance(metric_flag, list):
        metric_list = metric_flag  # `metric_flag` is already a list
    else:
        metric_list = metric_flag.split(',')  # Split `metric_flag` into a list
    for name in metric_list:
        try:
            # Now we can just look up the metrics in the dictionary we created
            metric = dict_of_metrics[name]
        except KeyError:  # No metric found with the name `name`
            raise ValueError(f"Unknown metric \"{name}\"")
        metrics[name] = metric()
    return metrics


def update_metrics(metrics, features, labels, pred_mean):
    """Update metrics

    Args:
        metrics: a dictionary with the initialized metrics
        features: the input
        labels: the correct labels
        pred_mean: the predicted mean
    """
    for name, metric in metrics.items():
        metric.update(features, labels, pred_mean)


def record_metrics(metrics, summary_writer=None, step_counter=None):
    """Print the result or record it in the summary

    Args:
        metrics: a dictionary with the updated metrics
    """
    for metric in metrics.values():
        result = metric.result()
        print(f"{metric.name}: {result}")
        if summary_writer is not None and step_counter is not None:
            summary_writer.add_scalar(metric.name, result, step_counter)


class Metric:
    """Base class for metrics"""
    name = "empty_metric"

    def update(self, features, labels, pred_mean):
        """Update the metric based on the given input, label and prediction

        Args:
            features: the input
            labels: the correct labels
            pred_mean: the predicted mean
        Returns:
            update op if `is_eager` is False
        """
        pass

    def result(self):
        """Compute the result"""
        pass


class Rmse(Metric):
    """Root mean squared error"""
    name = "RMSE"

    def __init__(self):
        super().__init__()
        self.metric = AverageValueMeter()

    def update(self, features, labels, pred_mean):
        self.metric.add((pred_mean - labels)**2)

    def result(self):
        return np.sqrt(self.metric.mean)


class Mae(Metric):
    """Mean absolute error"""
    name = "MAE"

    def __init__(self):
        super().__init__()
        self.mean = AverageValueMeter()

    def update(self, features, labels, pred_mean):
        self.mean.add((pred_mean - labels).abs().mean())

    def result(self):
        return self.mean


class SoftAccuracy(Metric):
    """Accuracy for softmax output"""
    name = "soft_accuracy"

    def __init__(self):
        super().__init__()
        self.accuracy = ClassErrorMeter(accuracy=True)

    def update(self, features, labels, pred_mean):
        self.accuracy.add(pred_mean.argmax(dim=1), labels.argmax(dim=1))

    def result(self):
        return self.accuracy.value(k=1)


class LogisticAccuracy(SoftAccuracy):
    """Accuracy for output from the logistic function"""
    name = "logistic_accuracy"

    def update(self, features, labels, pred_mean):
        self.accuracy.add(pred_mean.ge(0.5).int(), labels.add(1).mul(0.5).int())


class PredictionRateY1S0(Mae):
    """Acceptance Rate, group 1"""
    name = "pred_rate_y1_s0"

    def update(self, features, labels, pred_mean):
        accepted = tft.gather_nd(tf.cast(pred_mean > 0.5, tf.float32),
                                 tf.where(tfm.equal(features['sensitive'], 0)))
        return self._return_and_store(self.mean(accepted))


class PredictionRateY1S1(Mae):
    """Acceptance Rate, group 2"""
    name = "pred_rate_y1_s1"

    def update(self, features, labels, pred_mean):
        accepted = tft.gather_nd(tf.cast(pred_mean > 0.5, tf.float32),
                                 tf.where(tfm.equal(features['sensitive'], 1)))
        return self._return_and_store(self.mean(accepted))


class BaseRateY1S0(Mae):
    """Base acceptance rate, group 1"""
    name = "base_rate_y1_s0"

    def update(self, features, labels, pred_mean):
        accepted = tft.gather_nd(labels, tf.where(tfm.equal(features['sensitive'], 0)))
        return self._return_and_store(self.mean(accepted))


class BaseRateY1S1(Mae):
    """Base acceptance rate, group 2"""
    name = "base_rate_y1_s1"

    def update(self, features, labels, pred_mean):
        accepted = tft.gather_nd(labels, tf.where(tfm.equal(features['sensitive'], 1)))
        return self._return_and_store(self.mean(accepted))


class PredictionOddsYYbar1S0(Mae):
    """Opportunity P(yhat=1|s,ybar=1), group 1"""
    name = "pred_odds_yybar1_s0"

    def update(self, features, labels, pred_mean):
        test_for_ybar1_s0 = tfm.logical_and(tfm.equal(features['ybar'], 1),
                                            tfm.equal(features['sensitive'], 0))
        accepted = tft.gather_nd(tf.cast(pred_mean > 0.5, tf.float32), tf.where(test_for_ybar1_s0))
        return self._return_and_store(self.mean(accepted))


class PredictionOddsYYbar1S1(Mae):
    """Opportunity P(yhat=1|s,ybar=1), group 2"""
    name = "pred_odds_yybar1_s1"

    def update(self, features, labels, pred_mean):
        test_for_ybar1_s1 = tfm.logical_and(tfm.equal(features['ybar'], 1),
                                            tfm.equal(features['sensitive'], 1))
        accepted = tft.gather_nd(tf.cast(pred_mean > 0.5, tf.float32), tf.where(test_for_ybar1_s1))
        return self._return_and_store(self.mean(accepted))


class BaseOddsYYbar1S0(Mae):
    """Opportunity P(y=1|s,ybar=1), group 1"""
    name = "base_odds_yybar1_s0"

    def update(self, features, labels, pred_mean):
        test_for_ybar1_s0 = tfm.logical_and(tfm.equal(features['ybar'], 1),
                                            tfm.equal(features['sensitive'], 0))
        accepted = tft.gather_nd(labels, tf.where(test_for_ybar1_s0))
        return self._return_and_store(self.mean(accepted))


class BaseOddsYYbar1S1(Mae):
    """Opportunity P(y=1|s,ybar=1), group 2"""
    name = "base_odds_yybar1_s1"

    def update(self, features, labels, pred_mean):
        test_for_ybar1_s1 = tfm.logical_and(tfm.equal(features['ybar'], 1),
                                            tfm.equal(features['sensitive'], 1))
        accepted = tft.gather_nd(labels, tf.where(test_for_ybar1_s1))
        return self._return_and_store(self.mean(accepted))


class PredictionOddsYYbar0S0(Mae):
    """Opportunity P(yhat=1|s,ybar=1), group 1"""
    name = "pred_odds_yybar0_s0"

    def update(self, features, labels, pred_mean):
        test_for_ybar0_s0 = tfm.logical_and(tfm.equal(features['ybar'], 0),
                                            tfm.equal(features['sensitive'], 0))
        accepted = tft.gather_nd(tf.cast(pred_mean < 0.5, tf.float32), tf.where(test_for_ybar0_s0))
        return self._return_and_store(self.mean(accepted))


class PredictionOddsYYbar0S1(Mae):
    """Opportunity P(yhat=1|s,ybar=1), group 2"""
    name = "pred_odds_yybar0_s1"

    def update(self, features, labels, pred_mean):
        test_for_ybar0_s1 = tfm.logical_and(tfm.equal(features['ybar'], 0),
                                            tfm.equal(features['sensitive'], 1))
        accepted = tft.gather_nd(tf.cast(pred_mean < 0.5, tf.float32), tf.where(test_for_ybar0_s1))
        return self._return_and_store(self.mean(accepted))


class BaseOddsYYbar0S0(Mae):
    """Opportunity P(y=1|s,ybar=1), group 1"""
    name = "base_odds_yybar0_s0"

    def update(self, features, labels, pred_mean):
        test_for_ybar0_s0 = tfm.logical_and(tfm.equal(features['ybar'], 0),
                                            tfm.equal(features['sensitive'], 0))
        accepted = tft.gather_nd(1 - labels, tf.where(test_for_ybar0_s0))
        return self._return_and_store(self.mean(accepted))


class BaseOddsYYbar0S1(Mae):
    """Opportunity P(y=1|s,ybar=1), group 2"""
    name = "base_odds_yybar0_s1"

    def update(self, features, labels, pred_mean):
        test_for_ybar0_s1 = tfm.logical_and(tfm.equal(features['ybar'], 0),
                                            tfm.equal(features['sensitive'], 1))
        accepted = tft.gather_nd(1 - labels, tf.where(test_for_ybar0_s1))
        return self._return_and_store(self.mean(accepted))


class PredictionOddsYhatY1S0(Mae):
    """Opportunity P(yhat=1|s,y=1), group 1"""
    name = "pred_odds_yhaty1_s0"

    def update(self, features, labels, pred_mean):
        test_for_y1_s0 = tfm.logical_and(tfm.equal(labels, 1), tfm.equal(features['sensitive'], 0))
        accepted = tft.gather_nd(tf.cast(pred_mean > 0.5, tf.float32), tf.where(test_for_y1_s0))
        return self._return_and_store(self.mean(accepted))


class PredictionOddsYhatY1S1(Mae):
    """Opportunity P(yhat=1|s,y=1), group 2"""
    name = "pred_odds_yhaty1_s1"

    def update(self, features, labels, pred_mean):
        test_for_y1_s1 = tfm.logical_and(tfm.equal(labels, 1), tfm.equal(features['sensitive'], 1))
        accepted = tft.gather_nd(tf.cast(pred_mean > 0.5, tf.float32), tf.where(test_for_y1_s1))
        return self._return_and_store(self.mean(accepted))


class PredictionOddsYhatY0S0(Mae):
    """Opportunity P(yhat=0|s,y=0), group 1"""
    name = "pred_odds_yhaty0_s0"

    def update(self, features, labels, pred_mean):
        test_for_y0_s0 = tfm.logical_and(tfm.equal(labels, 0), tfm.equal(features['sensitive'], 0))
        accepted = tft.gather_nd(tf.cast(pred_mean < 0.5, tf.float32), tf.where(test_for_y0_s0))
        return self._return_and_store(self.mean(accepted))


class PredictionOddsYhatY0S1(Mae):
    """Opportunity P(yhat=0|s,y=0), group 2"""
    name = "pred_odds_yhaty0_s1"

    def update(self, features, labels, pred_mean):
        test_for_y0_s1 = tfm.logical_and(tfm.equal(labels, 0), tfm.equal(features['sensitive'], 1))
        accepted = tft.gather_nd(tf.cast(pred_mean < 0.5, tf.float32), tf.where(test_for_y0_s1))
        return self._return_and_store(self.mean(accepted))


def mask_for(features, **kwargs):
    """Create a 'mask' that filters for certain values

    Args:
        features: a dictionary of tensors
        **kwargs: entries of the dictionary with the values, only the first two are used
    Returns:
        a mask
    """
    entries = list(kwargs.items())
    return tf.where(tfm.logical_and(tfm.equal(features[entries[0][0]], entries[0][1]),
                                    tfm.equal(features[entries[1][0]], entries[1][1])))
