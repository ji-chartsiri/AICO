import numpy as np
import pandas as pd

from src.aico import AICO
from src.baseline import Baseline
from src.conditional_baseline import KNNConditionalSampler
from src.score import neg_squared_loss


def _make_correlated_dataset(n=200, seed=0):
    rng = np.random.RandomState(seed)
    x1 = rng.normal(size=n)
    x2 = x1 + rng.normal(scale=0.1, size=n)
    x3 = rng.normal(size=n)
    x = pd.DataFrame(dict(X0=1.0, X1=x1, X2=x2, X3=x3))
    y = x1 + 0.5 * x2 + rng.normal(scale=0.1, size=n)
    return x, y


def _pred_linear(x):
    return x["X1"] + 0.5 * x["X2"]


def test_knn_conditional_sampler_runs_through_aico():
    x, y = _make_correlated_dataset()
    x_train, x_test = x.iloc[:150], x.iloc[150:]
    y_train, y_test = y[:150], y[150:]

    sampler = KNNConditionalSampler(n_neighbors=5, conditioning_cols=["X2", "X3"], random_state=0)
    baseline = Baseline(conditional_samplers={"X1": sampler})

    aico = AICO(
        x_train=x_train,
        y_train=y_train,
        pred_func=_pred_linear,
        score_func=neg_squared_loss,
        baseline=baseline,
        vars_ignored=["X0"],
        vars_discrete=[],
        vars_categorical=[],
    )

    aico.test(x_test=x_test, y_test=y_test)

    assert "X1" in aico.result["variable"].values
    assert not aico.result.empty

