"""FairGP tests"""
import pandas as pd
from pytest import approx

from ethicml.algorithms import run_blocking
from ethicml.algorithms.inprocess import InAlgorithmAsync
from ethicml.data import load_data, Toy
from ethicml.preprocessing import train_test_split

from models import GPyT, GPyTDemPar, GPyTEqOdds


def test_gpyt():
    """test gpyt"""
    train, test = train_test_split(load_data(Toy()))

    baseline: InAlgorithmAsync = GPyT(flags=dict(epochs=2, length_scale=2.4))
    dem_par: InAlgorithmAsync = GPyTDemPar(flags=dict(epochs=2, length_scale=0.05))
    eq_odds: InAlgorithmAsync = GPyTEqOdds(tpr=1.0, flags=dict(epochs=2, length_scale=0.05))

    assert baseline.name == "GPyT_in_True"
    predictions: pd.DataFrame = run_blocking(baseline.run_async(train, test))
    assert predictions.values[predictions.values == 1].shape[0] == approx(210, rel=0.1)
    assert predictions.values[predictions.values == -1].shape[0] == approx(190, rel=0.1)

    assert dem_par.name == "GPyT_dem_par_in_True"
    predictions = run_blocking(dem_par.run_async(train, test))
    assert predictions.values[predictions.values == 1].shape[0] == approx(182, rel=0.1)
    assert predictions.values[predictions.values == -1].shape[0] == approx(218, rel=0.1)

    assert eq_odds.name == "GPyT_eq_odds_in_True_tpr_1.0"
    predictions = run_blocking(eq_odds.run_async(train, test))
    assert predictions.values[predictions.values == 1].shape[0] == approx(201, rel=0.1)
    assert predictions.values[predictions.values == -1].shape[0] == approx(199, rel=0.1)
