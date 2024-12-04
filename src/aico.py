import pandas as pd
from functools import partial

from src.baseline import Baseline
from src.utils import process_vars, summary
from src.test import compute_sign_test, compute_conf_int
from src.score import neg_squared_loss

class AICO:
    def __init__(self, x_train, y_train, pred_func, pred_params=dict(), score_func=neg_squared_loss, alpha=0.05, baseline=Baseline(), ignored_vars=[], discrete_vars=[], categorical_vars=[]):
        """
        Initializes the Add-In COvariate (AICO) test, which is a model-agnostic significance test for supervised machine learning models.

        The main idea of the AICO test is to assess the significance of individual features by comparing the response score
        before and after adding a feature value to the baseline feature set.

        Parameters:
        - x_train (pd.DataFrame): Feature data used for constructing baseline.
        - y_train (pd.Series or np.array): Response data used for constructing baseline.
        - pred_func (callable): Prediction function of the model.
        - pred_params (dict): Parameters for the prediction function.
        - score_func (callable): Score function used to evaluate model performance.
        - alpha (float): Significance level for the test.
        - baseline (Baseline): Baseline object to compute baseline and treatment features.
        - ignored_vars (list): List of variables to ignore from the feature set.
        - discrete_vars (list): List of discrete variables in the feature set.
        - categorical_vars (list or dict): If list, it should be the prefix of variables (e.g., "color" will consider
          columns like ["color_blue", "color_yellow"]). If dict, it should be a mapping from categorical variable name
          to the list of corresponding dummy variables (e.g., {"color": ["color_blue", "color_yellow"]}).
        """
        continuous_vars, discrete_vars, categorical_vars, self.all_vars = process_vars(x_train.columns, ignored_vars, discrete_vars, categorical_vars)
        self.pred_func = partial(pred_func, **pred_params)
        self.score_func = score_func
        self.alpha = alpha
        self.baseline = baseline
        self.baseline.update(x_train, y_train, pred_func, continuous_vars, discrete_vars, categorical_vars)

    def test(self, x_test, y_test):
        """
        Perform the AICO test on the test data.

        Parameters:
        - x_test (pd.DataFrame): Feature data used for testing.
        - y_test (pd.Series or np.array): Response data used for testing.
        """
        self.x_test = x_test
        self.y_test = y_test

        self.compute_response()     # compute f(\oX) and f(\uX)
        self.compute_delta()        # compute \Delta
        self.compute_test()         # perform test

    def summary(self):
        """
        Print a summary of the AICO test results.
        """
        summary(self.result, self.alpha, self.y_test.size, self.score_func)

    def compute_response(self):
        """
        Compute the baseline and treatment responses for each feature.
        """
        self.response_baseline = pd.DataFrame(index=self.x_test.index)
        self.response_treatment = self.response_baseline.copy()
        for var in self.all_vars[self.all_vars['type'] != 'ignored'].index:
            x_baseline, x_treatment = self.baseline(self.x_test, self.y_test, var)
            self.response_baseline[var] = self.pred_func(x_baseline)
            self.response_treatment[var] = self.pred_func(x_treatment)

    def compute_delta(self):
        """
        Compute the difference in score between the treatment and baseline responses.
        """
        self.delta = pd.DataFrame(index=self.x_test.index)
        for var in self.all_vars[self.all_vars['type'] != 'ignored'].index:
            self.delta[var] = self.score_func(self.response_treatment[var], self.y_test) - self.score_func(self.response_baseline[var], self.y_test)

    def compute_test(self):
        """
        Compute the p-values and confidence intervals for each feature and summarize the results.
        """
        p_value = pd.concat([compute_sign_test(self.delta[col], self.alpha) for col in self.delta.columns])
        conf_int = pd.concat([compute_conf_int(self.delta[col], self.alpha) for col in self.delta.columns])
        self.result = pd.merge(p_value, conf_int, left_index=True, right_index=True)
        self.result = (pd.merge(self.all_vars, self.result, left_index=True, right_index=True, how='left')
                       .reset_index()
                       .assign(sample_size=self.y_test.size,
                               score_func=self.score_func.__name__))
        self.result['rank'] = (self.result
                               .groupby('significance')
                               ['median']
                               .rank(ascending=False))

    def update(self, x_test=None, y_test=None, score_func=None, alpha=None):
        """
        Update the AICO test parameters and recompute if needed.

        Parameters:
        - x_test (pd.DataFrame, optional): Updated feature data for testing.
        - y_test (pd.Series or np.array, optional): Updated response data for testing.
        - score_func (callable, optional): Updated score function.
        - alpha (float, optional): Updated significance level.
        """
        self.x_test = self.x_test if x_test is None else x_test
        self.y_test = self.y_test if y_test is None else y_test
        self.score_func = self.score_func if score_func is None else score_func
        self.alpha = self.alpha if alpha is None else alpha

        if x_test is not None or y_test is not None:
            self.compute_response()
        if x_test is not None or y_test is not None or score_func is not None:
            self.compute_delta()
        if x_test is not None or y_test is not None or score_func is not None or alpha is not None:
            self.compute_test()
    