import numpy as np
import pandas as pd
from scipy.stats import binom
from statsmodels.stats.proportion import binom_test

def compute_sign_test(delta_var, alpha):
    """
    Compute the one-sided sign test for the given delta variable.

    Parameters:
    - delta_var (pd.Series): Series containing the delta values (differences in scores).
    - alpha (float): Significance level for the test.

    Returns:
    - result (pd.DataFrame): DataFrame containing p-values and significance flags for the given variable.
    """
    r_plus = (delta_var > 0).sum()
    result = pd.DataFrame(dict(p_value=binom_test(r_plus, delta_var.size, 0.5, 'larger'),
                               alpha=alpha),
                          index=[delta_var.name])
    result['significance'] = result['p_value'] <= result['alpha']
    return result

def compute_conf_int(delta_var, alpha):
    """
    Compute the confidence interval for the given delta variable.

    Parameters:
    - delta_var (pd.Series): Series containing the delta values (differences in scores).
    - alpha (float): Significance level for the test.

    Returns:
    - result (pd.DataFrame): DataFrame containing the median, lower, upper bounds, and coverage for the given variable.
    """
    n = delta_var.size
    delta_var = delta_var.copy().sort_values()
    median_var = delta_var.median()
    n_ci = int(binom.ppf(alpha/2, n, 0.5))
    if binom.cdf(n_ci, n, 0.5) > alpha/2:
        n_ci -= 1     # round-down to satisfy binom.cdf(n_ci, n, 0.5) <= alpha/2
    if n_ci < 0:
        n_lower = 0
        n_upper = n + 1
        lower = -np.inf
        upper = np.inf
        coverage = 1.0
    else:
        n_lower = 1 + n_ci
        n_upper = n - n_ci

        # zero-indexed
        lower = delta_var.iloc[n_lower - 1]   
        upper = delta_var.iloc[n_upper - 1]

        # handle ties (due to limited numerical precision)
        if lower == median_var:
            smaller_delta_var = delta_var[delta_var < lower]
            if smaller_delta_var.size > 0:
                lower = np.max(smaller_delta_var)
        if upper == median_var:
            larger_delta_var = delta_var[delta_var > upper]
            if larger_delta_var.size > 0:
                upper = np.min(larger_delta_var)

        coverage = 1 - binom.cdf(np.sum(delta_var > upper), n, 0.5) - binom.cdf(np.sum(delta_var < lower), n, 0.5)

    return pd.DataFrame(dict(median=median_var,
                             lower=lower,
                             upper=upper,
                             coverage=coverage),
                        index=[delta_var.name])
