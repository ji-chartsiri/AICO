import numpy as np
import pandas as pd

from keras.utils import set_random_seed

from scipy.stats import norm

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .utils import preprocess_intercept, preprocess_xy

def set_fig(num_subfig, suptitle, figsize=(30, 24), p=None):
    '''Setup the figure for multiple plots
    For p variables, suggest figsize = (30, 4 + 2.5*p)
    '''
    if p is not None:
        figsize = (30, 4 + 2.5*p)
    fig = plt.figure(layout='constrained', figsize=figsize)
    fig.suptitle(suptitle, fontweight='bold')
    return fig.subfigures(1, num_subfig, wspace=0.07, width_ratios=[1] * num_subfig)
    

def plot_data_hist(x, seed=0, bins=50, fig=None, subtitle=''):
    '''Plot the histogram of each X^k
    '''
    x, var_name = preprocess_xy(x)
    p = x.shape[1]
    axs = fig.subplots(p, 1, sharey='col')
    fig.suptitle('\n'.join(['[Distribution of $X^k$]',
                            subtitle]))
    for k in range(p):
        ax = axs[k]
        ax.hist(x[:, k], bins=bins)
        ax.set_title(f'{k}: {var_name[k]}')

def plot_delta_hist_fo(x, y, model, score_func, intercept, alpha=0.05, beta=None, num_beta_sim=50, beta_split_stratify=None,
                       seed=0, pred_params=dict(), bins=50, fig=None, subtitle=''):
    '''Plot the histogram of each delta^k. Due to performance reason, to avoid storing multiple sets of delta_k,
    the first-order AICO core test is also conducted in this function.
    '''
    from .test import compute_optimal_beta_fo, first_order_test_k

    set_random_seed(seed)

    # Pre-process arguments
    intercept = preprocess_intercept(intercept)
    x, var_name, y = preprocess_xy(x, y)
    p = x.shape[1]

    # Compute optimal beta
    if beta is None:
        beta = compute_optimal_beta_fo(x=x,
                                       y=y,
                                       model=model,
                                       score_func=score_func,
                                       intercept=intercept,
                                       alpha=alpha,
                                       num_beta_sim=num_beta_sim,
                                       beta_split_stratify=beta_split_stratify,
                                       seed=seed,
                                       pred_params=pred_params)

    if fig is not None:
        axs = fig.subplots(p, 1, sharey='col')
        fig.suptitle('\n'.join(['[Distribution of $Δ^k$]',
                                subtitle]))

    # Consider each variable (including the first variable which could be, but not necessary, intercept)
    results = []
    for k in range(p):
        result, delta_k = first_order_test_k(x=x,
                                             k=k, 
                                             var_name=var_name, 
                                             y=y, 
                                             model=model, 
                                             score_func=score_func, 
                                             intercept=intercept, 
                                             alpha=alpha, 
                                             beta=beta,
                                             seed=seed, 
                                             pred_params=pred_params)
        median_k, p_sign_test = result['median'], result['p_sign_test']
        results.append(result)
        
        if fig is not None:
            # Plot histogram of delta
            ## Some outliers are removed from the plot for clean visualization. This does not affect the test result and confidence interval obtained above.
            ax = axs[k]
            
            xabs_thres = np.quantile(np.abs(delta_k), 0.975)
            ax.hist(delta_k[(delta_k >= -xabs_thres) & (delta_k <= xabs_thres)], bins)
            ax.axvline(0, label=f'Zero', color='black', alpha=0.5)
            ax.axvline(median_k, label=f'Median = {median_k:.3E}', color='red', alpha=0.5)

            mean_k = np.mean(delta_k)
            ax.axvline(mean_k , label=f'Mean = {mean_k :.3E}', color='green', alpha=0.5)
            ax.set_title(f'{k}: {var_name[k]}')
            ax.legend(loc='upper right')
            ax.set_xlim(-xabs_thres - 0.05, xabs_thres + 0.05)
            ax.text(0.01, 0.97,
                    '\n'.join(['[p-value]',
                               '$H_0$: median($Δ^j$) = 0 | $H_1$: median($Δ^j$) > 0',
                               f'sign-test p-value: {p_sign_test:.5f} {"<significant>" if p_sign_test < alpha else ""}',
                               '',
                               f'Esimated P(Δ > 0) = {(delta_k > 0).mean():.5f}']),
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round',
                              facecolor='white',
                              alpha=0.5))
            
    results = pd.DataFrame(results)

    return results

def plot_ci_fo(results, fig=None, subtitle='', ci_range=None):
    '''Plot the confidence interval of median(delta^k)
    Assume results is for only one seed, alpha are the same for all rows, and k is zero-indexed consecutive
    '''
    p = results['k'].nunique()
    alpha = results['alpha'].iloc[0]

    axs = fig.subplots(p, 1, sharey='col')
    fig.suptitle('\n'.join(['[Confidence Interval of $Δ^k$]',
                            subtitle]))
    
    for k in range(p):
        ax = axs[k]
        result = results[results['k'] == k].iloc[0].to_dict()
        median_k, lower, upper, p_sign_test, var_name_k = result['median'], result['lower'], result['upper'], result['p_sign_test'], result['var_name_k']


        ax.errorbar([median_k], [0],
                    xerr=[[median_k - lower], [upper - median_k]],
                    fmt='o',
                    color='tab:red')
        ax.axvline(0, color='black', alpha=0.5)
        ax.set_title(f'{k}: {var_name_k} [{lower:.03g}, {upper:.03g}] {"<significant>" if p_sign_test < alpha else ""}')
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_alpha(0.5)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_ticks([])
        ax.xaxis.label.set_alpha(0.5)
        ax.tick_params(axis='x', which=u'both',length=3, color=[0,0,0,0.5])

    # Adjust range of x of CI plot
    x_min, x_max = results['lower'].min(), results['upper'].max()
    for k in range(p):
        ax = axs[k]
        if ci_range is None:
            ax.set_xlim(x_min, x_max)
        else:
            ax.set_xlim(ci_range[0], ci_range[1])
    

def plot_data_fo(x, y, model, intercept, zoom_y=False, pred_params=dict(), fig=None, subtitle=''):
    '''Plot the data and the predicted y across different X^k with other X's at their intercept
    '''
    # Pre-process arguments
    intercept = preprocess_intercept(intercept)
    x, var_name, y = preprocess_xy(x, y)
    p = x.shape[1]

    axs = fig.subplots(p, 1)
    fig.suptitle('\n'.join(['[$X^k$ vs Y and Predicted Y]',
                            subtitle]))
    
    for k in range(p):
        ax = axs[k]

        hb = ax.hexbin(x[:, k],
                       y,
                       gridsize=80,
                       norm=LogNorm(vmin=1,
                                    vmax=x.shape[0] // 100),
                       alpha=0.75)
        fig.colorbar(hb, ax=ax, label=f'count(data)')

        x_range_k = np.linspace(x[:, k].min(), x[:, k].max(), 1000)
        x_range = np.zeros((x_range_k.shape[0], x.shape[1]))
        x_range[:, k] = x_range_k

        x_corr_intercept = intercept(x_range, k)
        y_corr_intercept = model.predict(x_corr_intercept, **pred_params)

        x_corr_k = np.copy(x_corr_intercept)
        x_corr_k[:, k] = x_range_k
        y_corr_k = model.predict(x_corr_k, **pred_params)

        ax.plot(x_range_k,
                y_corr_intercept,
                label='Intercept',
                color='green',
                linewidth=2)
        
        ax.plot(x_range_k,
                y_corr_k,
                label='Partial Prediction',
                color='orange',
                linewidth=2)
        
        xabs_max = np.max(np.abs(x[:, k]))
        ax.set_xlim(-1.25*xabs_max, 1.25*xabs_max)
        if zoom_y:
            y_height = y_corr_k.max() - y_corr_k.min()
            ax.set_ylim(y_corr_k.min() - 0.5 * y_height,
                        y_corr_k.max() + 0.5 * y_height)

        ax.set_ylabel('Y')
        ax.set_title(f'{k}: {var_name[k]}')
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        
        '''
        ax.text(0.01, 0.97,
                '\n'.join(['[Intercept]',
                           f'Diag = {intercept[k, k]:.5f}',
                           f'Off-Diag = {x_corr_intercept[]:.5f}']),
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=props)
        '''
        ax.legend(loc='upper right')
