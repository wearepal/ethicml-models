"""Defines methods for metrics"""

import numpy as np
import torch
from torchnet.meter import AverageValueMeter, ClassErrorMeter


class Metric:
    """Base class for metrics"""
    name = "empty_metric"

    def update(self, inputs, labels, pred_mean):
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


class Mae(Metric):
    """Mean absolute error"""
    name = "MAE"

    def __init__(self):
        super().__init__()
        self.sum = 0.0
        self.num = 0.0

    def update(self, inputs, labels, pred_mean):
        self._add((pred_mean - labels).abs())

    def _add(self, values):
        values = values.cpu().float().numpy()
        self.sum += values.sum()
        self.num += values.shape[0]

    def result(self):
        return self.sum / self.num


class Rmse(Mae):
    """Root mean squared error"""
    name = "RMSE"

    def update(self, inputs, labels, pred_mean):
        self._add((pred_mean - labels)**2)

    def result(self):
        return np.sqrt(self.sum / self.num)


class ClassAccuracy(Mae):
    """Accuracy for softmax output"""
    name = "class_accuracy"

    def update(self, inputs, labels, pred_mean):
        self._add(torch.eq(pred_mean.argmax(dim=1), labels.argmax(dim=1)))


class BinaryAccuracy(Mae):
    """Accuracy for output from the logistic function"""
    name = "binary_accuracy"

    def update(self, inputs, labels, pred_mean):
        target, _ = torch.unbind(labels, dim=-1)
        accurate = torch.eq(pred_mean.ge(0.5).int(), target.add(1).mul(0.5).int()).float()
        self._add(accurate)


class PredictionRateY1S0(Mae):
    """Acceptance Rate, group 1"""
    name = "pred_rate_y1_s0"

    def update(self, inputs, labels, pred_mean):
        _, sens_attr = torch.unbind(labels, dim=-1)
        pred_s0 = torch.masked_select(pred_mean, torch.eq(sens_attr, 0))
        self._add(pred_s0.ge(0.5))


class PredictionRateY1S1(Mae):
    """Acceptance Rate, group 2"""
    name = "pred_rate_y1_s1"

    def update(self, inputs, labels, pred_mean):
        _, sens_attr = torch.unbind(labels, dim=-1)
        pred_s1 = torch.masked_select(pred_mean, torch.eq(sens_attr, 1))
        self._add(pred_s1.ge(0.5))


class BaseRateY1S0(Mae):
    """Base acceptance rate, group 1"""
    name = "base_rate_y1_s0"

    def update(self, inputs, labels, pred_mean):
        target, sens_attr = torch.unbind(labels, dim=-1)
        target_s0 = torch.masked_select(target, torch.eq(sens_attr, 0))
        self._add(target_s0.float().add(1).mul(0.5))


class BaseRateY1S1(Mae):
    """Base acceptance rate, group 2"""
    name = "base_rate_y1_s1"

    def update(self, inputs, labels, pred_mean):
        target, sens_attr = torch.unbind(labels, dim=-1)
        target_s1 = torch.masked_select(target, torch.eq(sens_attr, 1))
        self._add(target_s1.float().add(1).mul(0.5))


# class PredictionOddsYYbar1S0(Mae):
#     """Opportunity P(yhat=1|s,ybar=1), group 1"""
#     name = "pred_odds_yybar1_s0"

#     def update(self, inputs, labels, pred_mean):
#         test_for_ybar1_s0 = tfm.logical_and(tfm.equal(features['ybar'], 1),
#                                             tfm.equal(features['sensitive'], 0))
#         accepted = tft.gather_nd(tf.cast(pred_mean > 0.5, tf.float32), tf.where(test_for_ybar1_s0)
#         return self._return_and_store(self.mean(accepted))


# class PredictionOddsYYbar1S1(Mae):
#     """Opportunity P(yhat=1|s,ybar=1), group 2"""
#     name = "pred_odds_yybar1_s1"

#     def update(self, inputs, labels, pred_mean):
#         test_for_ybar1_s1 = tfm.logical_and(tfm.equal(features['ybar'], 1),
#                                             tfm.equal(features['sensitive'], 1))
#         accepted = tft.gather_nd(tf.cast(pred_mean > 0.5, tf.float32), tf.where(test_for_ybar1_s1)
#         return self._return_and_store(self.mean(accepted))


# class BaseOddsYYbar1S0(Mae):
#     """Opportunity P(y=1|s,ybar=1), group 1"""
#     name = "base_odds_yybar1_s0"

#     def update(self, inputs, labels, pred_mean):
#         test_for_ybar1_s0 = tfm.logical_and(tfm.equal(features['ybar'], 1),
#                                             tfm.equal(features['sensitive'], 0))
#         accepted = tft.gather_nd(labels, tf.where(test_for_ybar1_s0))
#         return self._return_and_store(self.mean(accepted))


# class BaseOddsYYbar1S1(Mae):
#     """Opportunity P(y=1|s,ybar=1), group 2"""
#     name = "base_odds_yybar1_s1"

#     def update(self, inputs, labels, pred_mean):
#         test_for_ybar1_s1 = tfm.logical_and(tfm.equal(features['ybar'], 1),
#                                             tfm.equal(features['sensitive'], 1))
#         accepted = tft.gather_nd(labels, tf.where(test_for_ybar1_s1))
#         return self._return_and_store(self.mean(accepted))


# class PredictionOddsYYbar0S0(Mae):
#     """Opportunity P(yhat=1|s,ybar=1), group 1"""
#     name = "pred_odds_yybar0_s0"

#     def update(self, inputs, labels, pred_mean):
#         test_for_ybar0_s0 = tfm.logical_and(tfm.equal(features['ybar'], 0),
#                                             tfm.equal(features['sensitive'], 0))
#         accepted = tft.gather_nd(tf.cast(pred_mean < 0.5, tf.float32), tf.where(test_for_ybar0_s0)
#         return self._return_and_store(self.mean(accepted))


# class PredictionOddsYYbar0S1(Mae):
#     """Opportunity P(yhat=1|s,ybar=1), group 2"""
#     name = "pred_odds_yybar0_s1"

#     def update(self, inputs, labels, pred_mean):
#         test_for_ybar0_s1 = tfm.logical_and(tfm.equal(features['ybar'], 0),
#                                             tfm.equal(features['sensitive'], 1))
#         accepted = tft.gather_nd(tf.cast(pred_mean < 0.5, tf.float32), tf.where(test_for_ybar0_s1)
#         return self._return_and_store(self.mean(accepted))


# class BaseOddsYYbar0S0(Mae):
#     """Opportunity P(y=1|s,ybar=1), group 1"""
#     name = "base_odds_yybar0_s0"

#     def update(self, inputs, labels, pred_mean):
#         test_for_ybar0_s0 = tfm.logical_and(tfm.equal(features['ybar'], 0),
#                                             tfm.equal(features['sensitive'], 0))
#         accepted = tft.gather_nd(1 - labels, tf.where(test_for_ybar0_s0))
#         return self._return_and_store(self.mean(accepted))


# class BaseOddsYYbar0S1(Mae):
#     """Opportunity P(y=1|s,ybar=1), group 2"""
#     name = "base_odds_yybar0_s1"

#     def update(self, inputs, labels, pred_mean):
#         test_for_ybar0_s1 = tfm.logical_and(tfm.equal(features['ybar'], 0),
#                                             tfm.equal(features['sensitive'], 1))
#         accepted = tft.gather_nd(1 - labels, tf.where(test_for_ybar0_s1))
#         return self._return_and_store(self.mean(accepted))


class PredictionOddsYhatY1S0(Mae):
    """Opportunity P(yhat=1|s,y=1), group 1"""
    name = "pred_odds_yhaty1_s0"

    def update(self, inputs, labels, pred_mean):
        target, sens_attr = torch.unbind(labels, dim=-1)
        test_for_y1_s0 = torch.eq(target, 1) * torch.eq(sens_attr, 0)
        accepted = torch.masked_select(pred_mean, test_for_y1_s0).ge(.5).float()
        self._add(accepted)


class PredictionOddsYhatY1S1(Mae):
    """Opportunity P(yhat=1|s,y=1), group 2"""
    name = "pred_odds_yhaty1_s1"

    def update(self, inputs, labels, pred_mean):
        target, sens_attr = torch.unbind(labels, dim=-1)
        test_for_y1_s1 = torch.eq(target, 1) * torch.eq(sens_attr, 1)
        accepted = torch.masked_select(pred_mean, test_for_y1_s1).ge(.5).float()
        self._add(accepted)


class PredictionOddsYhatY0S0(Mae):
    """Opportunity P(yhat=0|s,y=0), group 1"""
    name = "pred_odds_yhaty0_s0"

    def update(self, inputs, labels, pred_mean):
        target, sens_attr = torch.unbind(labels, dim=-1)
        test_for_y0_s0 = torch.eq(target, 0) * torch.eq(sens_attr, 0)
        accepted = torch.masked_select(pred_mean, test_for_y0_s0).ge(.5).float()
        self._add(accepted)


class PredictionOddsYhatY0S1(Mae):
    """Opportunity P(yhat=0|s,y=0), group 2"""
    name = "pred_odds_yhaty0_s1"

    def update(self, inputs, labels, pred_mean):
        target, sens_attr = torch.unbind(labels, dim=-1)
        test_for_y0_s1 = torch.eq(target, 0) * torch.eq(sens_attr, 1)
        accepted = torch.masked_select(pred_mean, test_for_y0_s1).ge(.5).float()
        self._add(accepted)
