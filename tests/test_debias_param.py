import numpy as np

from fairgp.fair_likelihood import (debiasing_params_target_rate, debiasing_params_target_tpr,
                                    positive_label_likelihood, debiasing_params_target_calibration)


RTOL = 1e-4
ATOL = 1e-10


class Namespace:
    pass


def rate_args(biased_rate1, biased_rate2, target_rate1, target_rate2,
              p_ybary0_or_ybary1_s0=1.0, p_ybary0_or_ybary1_s1=1.0):
    flags = Namespace()
    flags.target_rate1 = target_rate1
    flags.target_rate2=target_rate2
    flags.biased_acceptance1 = biased_rate1
    flags.biased_acceptance2 = biased_rate2
    flags.probs_from_flipped = False
    flags.p_ybary0_or_ybary1_s0 = p_ybary0_or_ybary1_s0
    flags.p_ybary0_or_ybary1_s1 = p_ybary0_or_ybary1_s1
    return flags


def tpr_args(biased_acceptance1, biased_acceptance2, p_ybary0_s0, p_ybary1_s0, p_ybary0_s1,
             p_ybary1_s1):
    flags = Namespace()
    flags.p_ybary0_s0 = p_ybary0_s0
    flags.p_ybary0_s1 = p_ybary0_s1
    flags.p_ybary1_s0 = p_ybary1_s0
    flags.p_ybary1_s1 = p_ybary1_s1
    flags.biased_acceptance1 = biased_acceptance1
    flags.biased_acceptance2 = biased_acceptance2
    return flags


def calibration_args(p_yybar0_s0, p_yybar1_s0, p_yybar0_s1, p_yybar1_s1):
    flags = Namespace()
    flags.p_yybar0_s0 = p_yybar0_s0
    flags.p_yybar0_s1 = p_yybar0_s1
    flags.p_yybar1_s0 = p_yybar1_s0
    flags.p_yybar1_s1 = p_yybar1_s1
    return flags


def invert(probs):
    probs = np.array(probs)
    return np.reshape(np.stack((1 - probs, probs), 0), (4, 2))


def construct(*, p_y0_ybar0_s0, p_y1_ybar1_s0, p_y0_ybar0_s1, p_y1_ybar1_s1):
    return np.log(invert([[1 - p_y0_ybar0_s0, p_y1_ybar1_s0],
                          [1 - p_y0_ybar0_s1, p_y1_ybar1_s1]]))


class TestDebiasParams:
    @staticmethod
    def test_extreme1():
        actual = debiasing_params_target_rate(rate_args(0.7, 0.7, 0.7, 0.7)).numpy()
        correct = construct(p_y0_ybar0_s0=1.,
                            p_y1_ybar1_s0=1.,
                            p_y0_ybar0_s1=1.,
                            p_y1_ybar1_s1=1.)
        np.testing.assert_allclose(actual, correct, RTOL)

    @staticmethod
    def test_extreme2():
        actual = debiasing_params_target_rate(rate_args(0.5, 0.5, 1e-5, 1 - 1e-5)).numpy()
        correct = construct(p_y0_ybar0_s0=.5,
                            p_y1_ybar1_s0=1.,
                            p_y0_ybar0_s1=1.,
                            p_y1_ybar1_s1=.5)
        np.testing.assert_allclose(actual, correct, RTOL)

    @staticmethod
    def test_moderate1():
        actual = debiasing_params_target_rate(rate_args(0.3, 0.7, 0.5, 0.5)).numpy()
        correct = construct(p_y0_ybar0_s0=1.,
                            p_y1_ybar1_s0=.3 / .5,
                            p_y0_ybar0_s1=1 - (.7 - .5) / .5,
                            p_y1_ybar1_s1=1.)
        np.testing.assert_allclose(actual, correct, RTOL, ATOL)

    @staticmethod
    def test_precision_target():
        p_y1_s0 = .3
        p_y1_s1 = .9
        p_ybar1_s0 = .5
        p_ybar1_s1 = .6
        prec_s0 = .7
        prec_s1 = .8
        flags = rate_args(p_y1_s0, p_y1_s1, p_ybar1_s0, p_ybar1_s1, prec_s0, prec_s1)
        actual_lik = positive_label_likelihood(flags, [p_y1_s0, p_y1_s1], [p_ybar1_s0, p_ybar1_s1])
        np.testing.assert_allclose(actual_lik,
                                   [[(p_ybar1_s0 - prec_s0 * p_y1_s0) / (1 - p_y1_s0),
                                     1 - .8],
                                    [.7,
                                     (p_ybar1_s1 - (1 - prec_s1) * (1 - p_y1_s1)) / p_y1_s1]],
                                   RTOL)
        actual_full = debiasing_params_target_rate(flags).numpy()
        correct = construct(p_y0_ybar0_s0=1 - (1 - prec_s0) * p_y1_s0 / (1 - p_ybar1_s0),
                            p_y1_ybar1_s0=prec_s0 * p_y1_s0 / p_ybar1_s0,
                            p_y0_ybar0_s1=prec_s1 * (1 - p_y1_s1) / (1 - p_ybar1_s1),
                            p_y1_ybar1_s1=1 - (1 - prec_s1) * (1 - p_y1_s1) / p_ybar1_s1)
        print(actual_full)
        print(correct)
        np.testing.assert_allclose(actual_full, correct, RTOL)


class TestEqOddsParams:
    @staticmethod
    def test_extreme1():
        actual = debiasing_params_target_tpr(tpr_args(.3, .7, 1., 1., 1., 1.)).numpy()
        correct = construct(p_y0_ybar0_s0=1.,
                            p_y1_ybar1_s0=1.,
                            p_y0_ybar0_s1=1.,
                            p_y1_ybar1_s1=1.)
        np.testing.assert_allclose(actual, correct, RTOL)

    @staticmethod
    def test_extreme2():
        actual = debiasing_params_target_tpr(tpr_args(.25, .75, .5, .5, .0, .0)).numpy()
        correct = construct(p_y0_ybar0_s0=.75,
                            p_y1_ybar1_s0=.25,
                            p_y0_ybar0_s1=0.,
                            p_y1_ybar1_s1=0.)
        np.testing.assert_allclose(actual, correct, RTOL)

    @staticmethod
    def test_moderate1():
        actual = debiasing_params_target_tpr(tpr_args(.3, .7, .8, 1., .8, 1.)).numpy()
        correct = construct(p_y0_ybar0_s0=1.,
                            p_y1_ybar1_s0=.3 / (.3 + .2 * .7),
                            p_y0_ybar0_s1=1.,
                            p_y1_ybar1_s1=.7 / (.7 + .2 * .3))
        np.testing.assert_allclose(actual, correct, RTOL)

    @staticmethod
    def test_moderate2():
        actual = debiasing_params_target_tpr(tpr_args(.1, .7, .8, .6, .4, .5)).numpy()
        correct = construct(p_y0_ybar0_s0=.8 * (1 - .1) / (.8 * (1 - .1) + (1 - .6) * .1),
                            p_y1_ybar1_s0=.6 * .1 / (.6 * .1 + (1 - .8) * (1 - .1)),
                            p_y0_ybar0_s1=.4 * (1 - .7) / (.4 * (1 - .7) + (1 - .5) * .7),
                            p_y1_ybar1_s1=.5 * .7 / (.5 * .7 + (1 - .4) * (1 - .7)))
        np.testing.assert_allclose(actual, correct, RTOL)


class TestCalibration:
    @staticmethod
    def test_extreme1():
        actual = np.exp(debiasing_params_target_calibration(calibration_args(.1, .2, .3, .4)).numpy())
        correct = np.array([[.1, .8],
                            [.3, .6],
                            [.9, .2],
                            [.7, .4]])
        np.testing.assert_allclose(actual, correct, RTOL)
