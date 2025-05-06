import pandas as pd
from functools import partial

from src.baseline import Baseline
from src.utils import process_vars, summary
from src.test import compute_response, compute_delta, compute_test, compute_rank, realize
from src.score import neg_squared_loss
from src.plot import plot_conditional

class AICO:
    def __init__(self, x_train, y_train, pred_func, pred_params=dict(), score_func=neg_squared_loss, alpha=0.05, baseline=Baseline(), vars_ignored=[], vars_discrete=[], vars_categorical=[]):
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
        - vars_ignored (list): List of variables to ignore from the feature set.
        - vars_discrete (list): List of discrete variables in the feature set.
        - vars_categorical (list or dict): If list, it should be the prefix of variables (e.g., "color" will consider
          columns like ["color_blue", "color_yellow"]). If dict, it should be a mapping from categorical variable name
          to the list of corresponding dummy variables (e.g., {"color": ["color_blue", "color_yellow"]}).
        """
        self.vars = process_vars(x_train.columns, vars_ignored, vars_discrete, vars_categorical)
        self.pred_func = partial(pred_func, **pred_params)
        self.score_func = score_func
        self.alpha = alpha
        self.baseline = baseline
        self.baseline.update(x_train, y_train, pred_func, self.vars)
        self.conditions = None
        self.seed = None

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
        self.compute_rank()         # compute the rank within each significance group (signifcant, inconclusive, insignificant)

    def condition(self, conditions=None):
        """
        Apply condition to the test set to perform conditional AICO test

        Parameters:
        - condition (pd.Series or np.array): Boolean masks indicating inclusion (or exclusion) of each sample
        """
        self.conditions = conditions
        self.compute_test()
        self.realize()

    def realize(self, seed=None):
        """
        Realize the randomized test, p-value, and confidence interval.

        Parameters:
        - seed (int or None): if int, the seed used for realization. If None, the test, p-value, 
                              and confidence interval will be unrealized; i.e., reverting them back to unrealized.
        """
        realize(self, seed)
        self.compute_rank()

    def summary(self):
        """
        Print a summary of the AICO test results.
        """
        summary(self.result)

    def compute_response(self):
        """
        Compute the baseline and treatment responses for each feature.
        """
        compute_response(self)

    def compute_delta(self):
        """
        Compute the difference in score between the treatment and baseline responses.
        """
        compute_delta(self)

    def compute_test(self):
        """
        Compute the p-values and confidence intervals for each feature and summarize the results.
        """
        compute_test(self)
        
    def compute_rank(self):
        """
        Update the variables rankings for each significance group: significant, inconclusive (when test hasn't been realized), and insignificant
        """
        compute_rank(self)

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
        self.compute_test()
        self.realize()
        self.compute_rank()
    
    def plot_conditional(self, var, var_delta, save_path=None):
        plot_conditional(self, var, var_delta, save_path)
