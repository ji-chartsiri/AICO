import math
import numpy as np
import pandas as pd
from keras.utils import set_random_seed
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.initializers import glorot_normal

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from statsmodels.stats.proportion import binom_test
from scipy.stats import norm

from .plot import set_fig, plot_delta_hist_fo, plot_ci_fo, plot_data_fo


def sign_test(z, M=0):
    r_plus = (z > M).sum()
    return binom_test(r_plus, z.size, 0.5, 'larger')


def compute_optimal_beta_fo(x, y, model, score_func, intercept, alpha=0.05, num_beta_sim=50, beta_split_stratify=None, seed=0, pred_params=None):
    '''Compute the optimal regularization hyperparameter beta for the neural network model's first-order aico test
    '''
    print('== Finding optimal beta ==')
    beta = 1e-7
    mean_nr_significants = 1
    initial_weights = model.get_weights()
    k_eval = lambda placeholder: placeholder.numpy()

    x_val, _, y_val, _ = train_test_split(x, y, test_size=0.5, stratify=beta_split_stratify)
    
    while mean_nr_significants > alpha:
        beta = beta * 10
        print(f'Try beta = {beta}')
        nr_significants_per_sim = np.zeros(num_beta_sim)
        for i in range(num_beta_sim):
            new_weights = [k_eval(glorot_normal()(w.shape)) for w in initial_weights]
            model.set_weights(new_weights)
            opt_beta_results = plot_delta_hist_fo(x_val, y_val, model, score_func, intercept, alpha=0.05, beta=beta,
                                                  seed=seed, pred_params=pred_params)
            nr_significants_per_sim[i] = (opt_beta_results['p_sign_test'] < alpha).mean()

        mean_nr_significants = np.mean(nr_significants_per_sim)
    print(f'Optimal beta = {beta}')
    model.set_weights(initial_weights)
    return beta


def first_order_test_k(x, k, var_name, y, model, score_func, intercept, alpha=0.05, beta=None,
                       seed=0, pred_params=dict()):
    '''Perform AICO test for a particular X_k
    '''
    n, p = x.shape[0], x.shape[1]
    x_intercept = intercept(x, k)
    y_intercept = model.predict(x_intercept, **pred_params)
     
    score_baseline = score_func(y_intercept, y)

    x_k = np.copy(x_intercept)
    x_k[:, k] = x[:, k]
    y_k = model.predict(x_k, **pred_params)
    score_k = score_func(y_k, y)

    delta_k = np.sort(score_k - (1-beta)*score_baseline)

    # p-value
    p_sign_test = sign_test(delta_k)

    # Confidence Interval
    median_k = np.median(delta_k)

    q = norm.ppf(1 - (alpha / 2))
    n_lower = math.floor(((n + 1) / 2 - q * math.sqrt(n) / 2))
    n_upper = math.ceil(((n + 1) / 2 + q * math.sqrt(n) / 2))
    lower = delta_k[n_lower]
    upper = delta_k[n_upper]

    result = dict(seed=seed,
                  k=k,
                  var_name_k=var_name[k],
                  score_func=score_func.__name__,
                  sample_size=n,
                  median=median_k,
                  p_sign_test=p_sign_test,
                  alpha=alpha,
                  lower=lower,
                  upper=upper)
    
    return result, delta_k


def first_order_test(x, y, model, score_func, intercept, alpha=0.05, beta=None, num_beta_sim=50, beta_split_stratify=None,
                     seed=0, visualize=True, pred_params=dict(), bins=50):
    '''Perform AICO test to test if each variable X_k is first-order significant

    Args:
        x (numpy.ndarray or pandas.DataFrame): tabular data of predictor wherein each row is each observation and each column is each X_k
        y (numpy.ndarray): vector of response
        model (object): model with respect to which the significance of each X_k will be tested. predict() method must be implemented.
        score_func (function): score function of format f(predicted y, original y) that returns a real number
        intercept (numpy.ndarray or Callable): the intercept values for first-order aico test
            - one-dimensional numpy.ndarray: the vector will be used as a common intercept for all tests (i.e., every test uses the same intercept)
            - two-dimensional numpy.ndarray: the k-th row of the array will be used as the intercept for x_k test
            - Callable: the function that inputs x (the data) and k (indicate the k-th test is being performed) 
                        and returns two-dimensional numpy.ndarray where i-th row is the intercept correspoding to the i-th sample
        alpha (float): significance level (from 0 to 1)
        seed (int): seed
        visualize (boolean): indicator whether visualization is needed
        pred_params (dictionary): the parameters to be passed to model.predict()
        bins (int): number of bins in delta histogram plot
    
    Returns:
        pandas.DataFrame: the first-order test results and confidence intervals.
    '''

    if visualize:
        figs = set_fig(3,
                   '\n'.join(['[AICO First-Order Test]',
                              f'score function = {score_func.__name__} | seed = {seed} | sample size = {x.shape[0]:,} | alpha = {alpha}']),
                              p=x.shape[1])
        
    results = plot_delta_hist_fo(x=x, 
                                 y=y, 
                                 model=model, 
                                 score_func=score_func, 
                                 intercept=intercept,
                                 alpha=alpha, 
                                 beta=beta, 
                                 num_beta_sim=num_beta_sim, 
                                 beta_split_stratify=beta_split_stratify,
                                 seed=seed,
                                 pred_params=pred_params,
                                 bins=bins,
                                 fig=figs[0] if visualize else None,
                                 subtitle='Mean Intercept')
    if visualize:
        plot_ci_fo(results=results,
                   fig=figs[1],
                   subtitle='Mean Intercept')

        plot_data_fo(x=x,
                     y=y,
                     model=model,
                     intercept=intercept,
                     pred_params=pred_params,
                     fig=figs[2],
                     subtitle='Mean Intercept')

        plt.show()

    return results


def check_second_order(x, y, model, score_func, intercept, first_order_result=None,
                      alpha=0.05, beta=None, seed=0, visualize=True, pred_params=dict(), bins=50):
    x = np.array(x)
    y = np.array(y)
    intercept = np.array(intercept)

    set_random_seed(seed)
    n, p = x.shape[0], x.shape[1]

    # Compute all-first-order score
    sig_var_1 = first_order_result[(first_order_result['seed'] == seed) &
                                   (first_order_result['p_sign_test'] < alpha)]['k']
    x_fo = np.zeros((n, p))
    x_fo[:, :] = intercept
    x_fo[:, sig_var_1] = x[:, sig_var_1]
    y_fo = model.predict(x_fo, **pred_params)
    score_fo = score_func(y_fo, y)

    # Compute full-x score
    y_full = model.predict(x, **pred_params)
    score_full = score_func(y_full, y)

    delta_fo = score_full - (1-beta)*score_fo

    # p-value
    p_sign_test = sign_test(delta_fo)

    median_fo = np.median(delta_fo)

    if visualize:
        xabs_thres = np.quantile(np.abs(delta_fo), 0.975)
        plt.figure(figsize=(12, 6))
        plt.hist(delta_fo[(delta_fo >= -xabs_thres) & (delta_fo <= xabs_thres)], bins)
        plt.axvline(0, label=f'Zero', color='black', alpha=0.5)
        plt.axvline(median_fo, label=f'Median = {median_fo:.3f}', color='red', alpha=0.5)
        plt.title('Check for higher-than-first-order effect')
        plt.legend()
        plt.xlim(-xabs_thres - 0.05, xabs_thres + 0.05)
        plt.text(0.01, 0.97,
                 '\n'.join(['[p-value]',
                            '$H_0$: median($Δ^{fo}$) = 0 | $H_1$: median($Δ^{fo}$) > 0',
                            f'sign-test p-value: {p_sign_test:.5f} {"<significant>" if p_sign_test < alpha else ""}'
                            ]),
                            transform=plt.gca().transAxes,
                            fontsize=10,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round',
                                        facecolor='white',
                                        alpha=0.5))
        plt.plot()

    return pd.DataFrame(dict(fo_k=[','.join([str(k) for k in sig_var_1])],
                             score_func=[score_func.__name__],
                             median=median_fo,
                             p_sign_test=p_sign_test,
                             alpha=alpha,
                             has_higher_order_effect=[p_sign_test < alpha]))
    

def second_order_test(x, y, model, score_func, intercept, k_list=None, first_order_result=None,
                      alpha=0.05, beta=0, seed=0, visualize=True, pred_params=dict(), bins=50):
    '''Perform AICO test to test if each pair (X_k, X_j) is second-order significant

    Args:
        x (numpy 2D array): matrix of predictor wherein each row is each observation and each column is each X_k
        y (numpy vector): vector of response
        model (object): model with respect to which the significance of each X_k will be tested. predict() method must be implemented.
        score_func (function): score function of format f(predicted y, original y) that returns a real number
        intercept (numpy vector): intercept values for each X_k
        k_list (list of int): list of k to be tested. If None, all X_k will be tested.
        first_order_result (Pandas DataFrame): result from first-order test. If None, a first-order test will be executed.
        alpha (float): significance level (from 0 to 1)
        seed (int): seed
        visualize (boolean): indicator whether visualization is needed
        pred_params (dictionary): the parameters to be passed to model.predict()
        bins (int): number of bins in delta histogram plot
    
    Returns:
        Pandas DataFrame: the second-order test results and confidence intervals.
    '''
    # Pre-process input
    if isinstance(x, np.ndarray):
        var_name = [f'X{k}' for k in range(p)]
    elif isinstance(x, pd.DataFrame):
        var_name = list(x.columns)
        x = np.array(x)
    else:
        raise Exception('x must be either Numpy array or Pandas dataframe')
    y = np.array(y)
    intercept = np.array(intercept)

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
            fig = plt.figure(layout='constrained', figsize=(30, 4 + 2.5*p))
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
                x_kj = x_j # Note: still refer to same object, essentially rename variable
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

            delta_kj = np.sort(score_kj - (1-beta)*score_baseline)

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
                                  var_name_k=var_name[k],
                                  j=j,
                                  var_name_j=var_name[j],
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
                ax_delta_hist.set_title(f'{k}, {j}: {var_name[k]}, {var_name[j]}')
                ax_delta_hist.legend()
                ax_delta_hist.set_xlim(-xabs_thres - 0.05, xabs_thres + 0.05)
                ax_delta_hist.text(0.01, 0.97,
                                   '\n'.join(['[p-value]',
                                              '$H_0$: median($Δ^{kj}$) = 0 | $H_1$: median($Δ^{kj}$) > 0',
                                              f'sign-test p-value: {p_sign_test:.5f} {"<significant>" if p_sign_test < alpha else ""}',
                                              f'Note: $X_{{{j}}}$ is first-order {"significant" if sig_var_1[j] else "insignificant"}'
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
                ax_delta_ci.set_title(f'{k}, {j}: {var_name[k]}, {var_name[j]} [{delta_kj[n_lower]:.03g}, {delta_kj[n_upper]:.03g}] {"<significant>" if p_sign_test < alpha else ""}')
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
            for j in range(p - 1):
                ax_delta_ci = axs_delta_ci[j]
                ax_delta_ci.set_xlim(x_min, x_max)

        plt.show()

    results = pd.concat(results)
    return results
