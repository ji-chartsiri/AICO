import numpy as np
from .aggregate import *

class Baseline:
    def __init__(self, continuous_agg=MeanAgg(), discrete_agg=AltModeAgg(), categorical_agg=AltModeAgg(), glob=True):
        self.continuous_agg = continuous_agg
        self.discrete_agg = discrete_agg
        self.categorical_agg = categorical_agg
        self.glob = glob
    
    def update(self, x_train, y_train, pred_func, vars):
        self.vars = vars
        self.agg = dict()
        for var_name, var in vars.iterrows():
            if var['type'] == 'continuous':
                agg = self.continuous_agg
            elif var['type'] == 'discrete':
                agg = self.discrete_agg
            elif var['type'] == 'categorical':
                agg = self.categorical_agg
            self.agg[var_name] = agg(x_train[var['columns']], y_train, pred_func)

    def __call__(self, x_test, y_test, test_var):
        x_baseline = x_test.copy()
        x_treatment = x_baseline.copy()

        test_cols = self.vars.loc[test_var]['columns']
        x_baseline[test_cols] = self.agg[test_var](x_test[test_cols], y_test)

        if not self.glob:
            for var_name, var in self.vars.iterrows():
                if var_name != test_var:
                    cols = var['columns']
                    x_baseline[cols] = self.agg[var_name](x_test[cols], y_test)
                    x_treatment[cols] = x_baseline[cols].copy()

        return x_baseline, x_treatment
