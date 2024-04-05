import math
import numpy as np
import pandas as pd
from keras.utils import set_random_seed

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from statsmodels.stats.proportion import binom_test
from scipy.stats import norm

def sign_test(z, M=0):
    r_plus = (z > M).sum()
    return binom_test(r_plus, z.size, 0.5, 'larger')

def first_order_test(x, y, model, score_func, intercept, alpha=0.05,
                     seed=0, visualize=True, pred_params=dict(), bins=50):
    '''Perform AICO test to test if each variable X_k is first-order significant

    Args:
        x (numpy 2D array): matrix of predictor wherein each row is each observation and each column is each X_k
        y (numpy vector): vector of response
        model (object): model with respect to which the significance of each X_k will be tested. predict() method must be implemented.
        score_func (function): score function of format f(predicted y, original y) that returns a real number
        intercept (numpy vector): intercept values for each X_k
        alpha (float): significance level (from 0 to 1)
        seed (int): seed
        visualize (boolean): indicator whether visualization is needed
        pred_params (dictionary): the parameters to be passed to model.predict()
        bins (int): number of bins in delta histogram plot
    
    Returns:
        Pandas DataFrame: the first-order test results and confidence intervals.
    '''
    set_random_seed(seed)
    n, p = x.shape[0], x.shape[1]

    # Compute baseline score
    x_intercept = np.zeros((n, p))
    x_intercept[:, :] = intercept
    y_intercept = model.predict(x_intercept, **pred_params)
    
    score_baseline = score_func(y_intercept, y)

    if visualize:
        fig = plt.figure(layout='constrained', figsize=(30, 4 + 2*p))
        fig.suptitle('\n'.join(['[AICO First-Order Test]',
                                f'score function = {score_func.__name__} | seed = {seed} | sample size = {n:,} | alpha = {alpha}']),
                     fontweight='bold')
        fig = fig.subfigures(1, 3, wspace=0.07, width_ratios=[1] * 3)

        # Delta histogram
        axs_delta_hist = fig[0].subplots(p , 1, sharey='col')
        fig[0].suptitle('[Distribution of $Δ^k$]')

        # Delta confidence interval
        axs_delta_ci = fig[1].subplots(p , 1, sharey='col')
        fig[1].suptitle('[Confidence Interval of $Δ^k$]')

        # Data plot
        axs_data = fig[2].subplots(p , 1, sharey='col')
        fig[2].suptitle('[$X_k$ vs Y and Predicted Y]')

    # Consider each variable (including the first variable which could be, but not necessary, intercept)
    results = []
    for k in range(p):
        x_k = np.copy(x_intercept)
        x_k[:, k] = x[:, k]
        y_k = model.predict(x_k, **pred_params)
        score_k = score_func(y_k, y)

        delta_k = np.sort(score_k - score_baseline)

        # p-value
        p_sign_test = sign_test(delta_k)

        # Confidence Interval
        median_k = np.median(delta_k)

        q = norm.ppf(1 - (alpha / 2))
        n_lower = math.floor(((n + 1) / 2 - q * math.sqrt(n) / 2))
        n_upper = math.ceil(((n + 1) / 2 + q * math.sqrt(n) / 2))
        lower = delta_k[n_lower]
        upper = delta_k[n_upper]

        results.append(dict(seed=seed,
                            k=k,
                            score_func=score_func.__name__,
                            sample_size=n,
                            median=median_k,
                            p_sign_test=p_sign_test,
                            alpha=alpha,
                            lower=lower,
                            upper=upper))
        
        if visualize:
            # Plot histogram of delta
            ## Some outliers are removed from the plot for clean visualization. This does not affect the test result and confidence interval obtained above.
            ax_delta_hist = axs_delta_hist[k]
            xabs_thres = np.quantile(np.abs(delta_k), 0.975)
            ax_delta_hist.hist(delta_k[(delta_k >= -xabs_thres) & (delta_k <= xabs_thres)], bins)
            ax_delta_hist.axvline(0, label=f'Zero', color='black', alpha=0.5)
            ax_delta_hist.axvline(median_k, label=f'Median = {median_k:.3f}', color='red', alpha=0.5)
            ax_delta_hist.set_title(f'$Δ^{{{k}}}$')
            ax_delta_hist.legend()
            ax_delta_hist.set_xlim(-xabs_thres - 0.05, xabs_thres + 0.05)
            ax_delta_hist.text(0.01, 0.97,
                               '\n'.join(['[p-value]',
                                          '$H_0$: median($Δ^j$) = 0 | $H_1$: median($Δ^j$) > 0',
                                          f'sign-test p-value: {p_sign_test:.5f} {"<significant>" if p_sign_test < alpha else ""}'
                                          ]),
                                transform=ax_delta_hist.transAxes,
                                fontsize=10,
                                verticalalignment='top',
                                bbox=dict(boxstyle='round',
                                          facecolor='white',
                                          alpha=0.5))
            
            # Plot confidence interval of delta
            ax_delta_ci = axs_delta_ci[k]
            ax_delta_ci.errorbar([median_k], [0],
                                 xerr=[[median_k - lower], [upper - median_k]],
                                 fmt='o',
                                 color='tab:red')
            ax_delta_ci.axvline(0, color='black', alpha=0.5)
            ax_delta_ci.set_title(f'$X_{{{k}}}$: [{delta_k[n_lower]:.03g}, {delta_k[n_upper]:.03g}] {"<significant>" if p_sign_test < alpha else ""}')
            ax_delta_ci.spines['top'].set_visible(False)
            ax_delta_ci.spines['bottom'].set_alpha(0.5)
            ax_delta_ci.spines['right'].set_visible(False)
            ax_delta_ci.spines['left'].set_visible(False)
            ax_delta_ci.get_yaxis().set_ticks([])
            ax_delta_ci.xaxis.label.set_alpha(0.5)
            ax_delta_ci.tick_params(axis='x', which=u'both',length=3, color=[0,0,0,0.5])

            # Plot data, intercept, and predicted Y from X with only X_k (other X's are at intercept)
            ax_data = axs_data[k]

            hb = ax_data.hexbin(x[:, k],
                                y,
                                gridsize=80,
                                norm=LogNorm(vmin=1,
                                             vmax=x.shape[0] // 100),
                                alpha=0.75)
            fig[2].colorbar(hb, ax=ax_data, label=f'count(data)')

            y_corr_intercept = model.predict(intercept.reshape(1, -1), **pred_params)

            xabs_max = np.max(np.abs(x[:, k]))
            x_range = np.linspace(-xabs_max, xabs_max, 1000)
            x_corr_k = np.zeros((x_range.size, p))
            x_corr_k[:, :] = intercept
            x_corr_k[:, k] = x_range
            y_corr_k = model.predict(x_corr_k, **pred_params)

            ax_data.axhline(y_corr_intercept,
                            label=f'Intercept',
                            color='green')
            
            ax_data.plot(x_range,
                         y_corr_k,
                         label='Partial Prediction',
                         color='orange',
                         linewidth=2)
            
            ax_data.set_xlim(-xabs_max, xabs_max)
            ax_data.set_ylabel('Y')
            ax_data.set_title(f'$X_{{k}}$')
            ax_data.legend()

    results = pd.DataFrame(results)

    if visualize:
        # Adjust range of x of CI plot
        x_min, x_max = results['lower'].min(), results['upper'].max()
        for k in range(p):
            ax_delta_ci = axs_delta_ci[k]
            ax_delta_ci.set_xlim(x_min, x_max)

    plt.show()

    return results


def second_order_test(x, y, model, score_func, intercept, k_list=None, first_order_result=None,
                      alpha=0.05, seed=0, visualize=True, pred_params=dict(), bins=50):
    set_random_seed(seed)
    n, p = x.shape[0], x.shape[1]

    if k_list is None:
        k_list = range(p)
        
    if first_order_result is None:
        first_order_result = first_order_test(x, y, model, score_func, intercept, alpha=alpha,
                                              seed=seed, visualize=False)
    
    # Find which variables are first-order significant
    sig_var_1 = (first_order_result[first_order_result['seed'] == seed]
                 .set_index('k')['p_sign_test'] < alpha).to_dict()
    
    # Compute baseline score
    x_intercept = np.zeros((n, p))
    x_intercept[:, :] = intercept

    # Consider each X_k
    results = []
    for k in k_list:
        if visualize:
            fig = plt.figure(layout='constrained', figsize=(30, 4 + 2*p))
            fig.suptitle('\n'.join([f'[AICO Second-Order Test | Pairs ($X_k$, $X_j$) with k = {k}]',
                                    f'score function = {score_func.__name__} | seed = {seed} | sample size = {n:,} | alpha = {alpha}']),
                        fontweight='bold')
            fig = fig.subfigures(1, 2, wspace=0.07, width_ratios=[1] * 2)

            # Delta histogram
            axs_delta_hist = fig[0].subplots(p - 1, 1, sharey='col')
            fig[0].suptitle('[Distribution of $Δ^{kj}$]')

            # Delta confidence interval
            axs_delta_ci = fig[1].subplots(p - 1, 1, sharey='col')
            fig[1].suptitle('[Confidence Interval of $Δ^{kj}$]')

        # Testing X_k by pairing with each X_j
        results_k = []
        for j in range(p):
            if k == j:
                continue
            elif sig_var_1[j]:
                # If X_j is significant
                x_j = np.copy(x_intercept)
                x_j[:, j] = x[:, j]
                y_j = model.predict(x_j, **pred_params)
                score_baseline = score_func(y_j, y)

                # Introduce X_k
                x_kj = x_j
                x_kj[:, k] = x[:, k]
            else:
                # If X_j is insignificant
                y_intercept = model.predict(x_intercept, **pred_params)
                score_baseline = score_func(y_intercept, y)

                # Introduce X_k and X_j
                x_kj = np.copy(x_intercept)
                x_kj[:, k] = x[:, k]
                x_kj[:, j] = x[:, j]

            y_kj = model.predict(x_kj, **pred_params)
            score_kj = score_func(y_kj, y)

            delta_kj = np.sort(score_kj - score_baseline)

            # p-value
            p_sign_test = sign_test(delta_kj)

            # Confidence Interval
            median_kj = np.median(delta_kj)

            q = norm.ppf(1 - (alpha / 2))
            n_lower = math.floor(((n + 1) / 2 - q * math.sqrt(n) / 2))
            n_upper = math.ceil(((n + 1) / 2 + q * math.sqrt(n) / 2))
            lower = delta_kj[n_lower]
            upper = delta_kj[n_upper]

            results_k.append(dict(seed=seed,
                                  k=k,
                                  j=j,
                                  j_first_order_sig=sig_var_1[j],
                                  score_func=score_func.__name__,
                                  sample_size=n,
                                  median=median_kj,
                                  p_sign_test=p_sign_test,
                                  alpha=alpha,
                                  lower=lower,
                                  upper=upper))
            
            if visualize:
                # Plot histogram of delta
                ## Some outliers are removed from the plot for clean visualization. This does not affect the test result and confidence interval obtained above.
                if j < k:
                    ax_delta_hist = axs_delta_hist[j]
                elif j > k:
                    ax_delta_hist = axs_delta_hist[j - 1]
                xabs_thres = np.quantile(np.abs(delta_kj), 0.975)
                ax_delta_hist.hist(delta_kj[(delta_kj >= -xabs_thres) & (delta_kj <= xabs_thres)], bins)
                ax_delta_hist.axvline(0, label=f'Zero', color='black', alpha=0.5)
                ax_delta_hist.axvline(median_kj, label=f'Median = {median_kj:.3f}', color='red', alpha=0.5)
                ax_delta_hist.set_title(f'$Δ^{{({k}, {j})}}$')
                ax_delta_hist.legend()
                ax_delta_hist.set_xlim(-xabs_thres - 0.05, xabs_thres + 0.05)
                ax_delta_hist.text(0.01, 0.97,
                                   '\n'.join(['[p-value]',
                                              '$H_0$: median($Δ^{kj}$) = 0 | $H_1$: median($Δ^{kj}$) > 0',
                                              f'sign-test p-value: {p_sign_test:.5f} {"<significant>" if p_sign_test < alpha else ""}'
                                              ]),
                                   transform=ax_delta_hist.transAxes,
                                   fontsize=10,
                                   verticalalignment='top',
                                   bbox=dict(boxstyle='round',
                                             facecolor='white',
                                             alpha=0.5))
                
                # Plot confidence interval of delta
                if j < k:
                    ax_delta_ci = axs_delta_ci[j]
                elif j > k:
                    ax_delta_ci = axs_delta_ci[j - 1]
                ax_delta_ci.errorbar([median_kj], [0],
                                    xerr=[[median_kj - lower], [upper - median_kj]],
                                    fmt='o',
                                    color='tab:red')
                ax_delta_ci.axvline(0, color='black', alpha=0.5)
                ax_delta_ci.set_title(f'$X_{{({k}, {j})}}$: [{delta_kj[n_lower]:.03g}, {delta_kj[n_upper]:.03g}] {"<significant>" if p_sign_test < alpha else ""}')
                ax_delta_ci.spines['top'].set_visible(False)
                ax_delta_ci.spines['bottom'].set_alpha(0.5)
                ax_delta_ci.spines['right'].set_visible(False)
                ax_delta_ci.spines['left'].set_visible(False)
                ax_delta_ci.get_yaxis().set_ticks([])
                ax_delta_ci.xaxis.label.set_alpha(0.5)
                ax_delta_ci.tick_params(axis='x', which=u'both',length=3, color=[0,0,0,0.5])

        results_k = pd.DataFrame(results_k)
        results.append(results_k)
        if visualize:
            # Adjust range of x of CI plot
            x_min, x_max = results_k['lower'].min(), results_k['upper'].max()
            for j in range(p):
                ax_delta_ci = axs_delta_ci[j]
                ax_delta_ci.set_xlim(x_min, x_max)

        plt.show()

    results = pd.concat(results)
    return results
