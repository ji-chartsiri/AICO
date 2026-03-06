import numpy as np
import pandas as pd

from src.multipletest import apply_fdr, apply_fwer


def _toy_result():
    variables = ["X1", "X2", "X3", "X4"]
    pvals = np.array([0.001, 0.01, 0.2, 0.5])
    return pd.DataFrame(
        dict(
            variable=variables,
            type=["continuous"] * 4,
            p_value=pvals,
            p_value_lower=pvals,
            p_value_upper=pvals,
        )
    ).set_index("variable")


def test_apply_fdr_bh_marks_small_p_as_significant():
    result = _toy_result()
    result_corr = apply_fdr(result, alpha=0.05, method="BH")

    assert result_corr.loc["X1", "significant_fdr"]
    assert result_corr.loc["X2", "significant_fdr"]
    assert not result_corr.loc["X3", "significant_fdr"]
    assert not result_corr.loc["X4", "significant_fdr"]


def test_apply_fwer_holm_is_conservative():
    result = _toy_result()
    result_corr = apply_fwer(result, alpha=0.05, method="holm")

    assert result_corr.loc["X1", "significant_fwer"]
    assert result_corr.loc["X2", "significant_fwer"]
    assert not result_corr.loc["X3", "significant_fwer"]
    assert not result_corr.loc["X4", "significant_fwer"]

