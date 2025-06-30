import numpy as np
import pandas as pd
from scipy.stats import binom
from statsmodels.stats.proportion import binom_test

def compute_response(aico):
    """
    Compute the baseline and treatment responses for each feature.
    """
    aico.response_baseline = dict()
    aico.response_treatment = aico.response_baseline.copy()
    for var in aico.vars[aico.vars['type'] != 'ignored'].index:
        x_baseline, x_treatment = aico.baseline(aico.x_test, aico.y_test, var)
        aico.response_baseline[var] = aico.pred_func(x_baseline)
        aico.response_treatment[var] = aico.pred_func(x_treatment)


def compute_delta(aico):
    """
    Compute the difference in score between the treatment and baseline responses.
    """
    aico.delta = pd.DataFrame(index=aico.x_test.index)
    for var in aico.vars[aico.vars['type'] != 'ignored'].index:
        aico.delta[var] = aico.score_func(aico.response_treatment[var], aico.y_test) - aico.score_func(aico.response_baseline[var], aico.y_test)


def compute_test(aico):
    """
    Compute the p-values and confidence intervals for each feature and summarize the results.
    """
    delta = aico.delta if aico.conditions is None else aico.delta[aico.conditions]
    delta = delta if aico.groups is None else delta.groupby(aico.groups if aico.conditions is None else aico.groups[aico.conditions]).mean()
    p_value = pd.concat([compute_sign_test(delta[col], aico.alpha) for col in delta.columns])
    conf_int = pd.concat([compute_conf_int(delta[col], aico.alpha) for col in delta.columns])
    aico.result = pd.merge(p_value, conf_int, left_index=True, right_index=True)
    aico.result = (pd.merge(aico.vars, aico.result, left_index=True, right_index=True, how='left')
                   .reset_index())
    aico.result.loc[aico.result['type'] != 'ignored', ['sample_size', 'score_func', 'seed']] = delta.shape[0], aico.score_func.__name__, None


def compute_rank(aico):
    """
    Update the variables rankings for each significance group: significant, inconclusive (when test hasn't been realized), and insignificant
    """
    aico.result['rank'] = (aico.result
                            .groupby('significance', dropna=False)
                            ['median']
                            .rank(ascending=False))


def realize(aico, seed):
    """
    Realize the randomized test, p-value, and confidence interval.

    Parameters:
    - seed (int or None): if int, the seed used for realization. If None, the test, p-value, 
                            and confidence interval will be unrealized; i.e., reverting them back to unrealized.
    """
    aico.seed = seed
    aico.result.loc[aico.result['type'] != 'ignored', 'seed'] = seed
    if seed is None:
        aico.result[['significance', 'p_value', 'lower_os']] = None
        aico.result.loc[aico.result['prob_reject'] == 0, 'significance'] = False
        aico.result.loc[aico.result['prob_reject'] == 1, 'significance'] = True
    else:
        U = np.random.uniform(0, 1, aico.result.shape[0])
        aico.result.loc[U < aico.result['prob_reject'], 'significance'] = True
        aico.result.loc[U >= aico.result['prob_reject'], 'significance'] = False
        aico.result['p_value'] = aico.result['p_value_lower'] + U * (aico.result['p_value_upper'] - aico.result['p_value_lower'])
        aico.result.loc[U < aico.result['prob_os'], 'lower_os'] = aico.result['lower_os_2']
        aico.result.loc[U >= aico.result['prob_os'], 'lower_os'] = aico.result['lower_os_1']


def compute_sign_test(delta_var, alpha):
    """
    Compute the one-sided sign test for the given delta variable.

    Parameters:
    - delta_var (pd.Series): Series containing the delta values (differences in scores).
    - alpha (float): Significance level for the test.

    Returns:
    - result (pd.DataFrame): DataFrame containing p-values and significance flags for the given variable.
    """
    n = delta_var.size
    r_plus = (delta_var > 0).sum()

    # test function
    threshold = binom.ppf(1.0 - alpha, n, 0.5)
    if r_plus > threshold:
        prob_reject = 1.0
        significance = True
    elif r_plus == threshold:
        prob_reject = (binom.cdf(r_plus, n, 0.5) - (1 - alpha)) / binom.pmf(r_plus, n, 0.5)
        significance = None
    else:
        prob_reject = 0.0
        significance = False

    # p-value
    p_value_lower = binom_test(r_plus + 1, delta_var.size, 0.5, 'larger')
    p_value_upper = binom_test(r_plus, delta_var.size, 0.5, 'larger')

    result = pd.DataFrame(dict(prob_reject=prob_reject,
                               significance=significance,
                               p_value=None,
                               p_value_lower=p_value_lower,
                               p_value_upper=p_value_upper,
                               alpha=alpha),
                          index=[delta_var.name])
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

    # two-sided confidence interval (non-randomized)
    C_ts = int(binom.ppf(1.0 - alpha/2, n, 0.5))
    n_lower_ts, n_upper_ts = (n - C_ts) - 1, C_ts

    lower_ts = -np.inf if n_lower_ts == -1 else delta_var.iloc[n_lower_ts]
    upper_ts = np.inf if n_upper_ts == n else delta_var.iloc[n_upper_ts]
    ## also handle case where ties are present
    lower_cdf = 0 if lower_ts == -np.inf else binom.cdf(np.sum(delta_var < lower_ts), n, 0.5)
    upper_cdf = 0 if upper_ts == np.inf else binom.cdf(np.sum(delta_var > upper_ts), n, 0.5)
    coverage_ts = 1 - lower_cdf - upper_cdf

    # one-sided confidence interval (unrealized)
    C_os = int(binom.ppf(1.0 - alpha, n, 0.5))
    n_lower_os = (n - C_os) - 1
    
    lower_os_1 = -np.inf if n_lower_os == 0 else delta_var.iloc[n_lower_os]
    lower_os_2 = np.inf if n_lower_os + 1 == n else delta_var.iloc[n_lower_os + 1]
    prob_os = (binom.cdf(C_os, n, 0.5) - (1 - alpha)) / binom.pmf(C_os, n, 0.5) # probability of getting more conservative CI (lower_os_2)

    return pd.DataFrame(dict(median=median_var,
                             lower_ts=lower_ts,
                             upper_ts=upper_ts,
                             coverage_ts=coverage_ts,
                             lower_os=None,
                             lower_os_1=lower_os_1,
                             lower_os_2=lower_os_2,
                             prob_os=prob_os,
                             coverage_os=1-alpha),
                             index=[delta_var.name])
