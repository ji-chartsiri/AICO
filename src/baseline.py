import numpy as np
from .aggregate import *

class Baseline:
    def __init__(self, continuous_agg=MeanAgg(), discrete_agg=AltModeAgg(), categorical_agg=AltModeAgg(), glob=True):
        self.continuous_agg = continuous_agg
        self.discrete_agg = discrete_agg
        self.categorical_agg = categorical_agg
        self.glob = glob
    
    def update(self, x_train, y_train, pred_func, continuous_vars, discrete_vars, categorical_vars):
        self.continuous_vars = continuous_vars
        self.discrete_vars = discrete_vars
        self.categorical_vars = categorical_vars

        self.agg = dict()
        for var in continuous_vars:
            self.agg[var] = self.continuous_agg(x_train[[var]], y_train, pred_func)
        for var in discrete_vars:
            self.agg[var] = self.discrete_agg(x_train[[var]], y_train, pred_func)
        for var in categorical_vars:
            dummy = categorical_vars[var]
            self.agg[var] = self.categorical_agg(x_train[dummy], y_train, pred_func)

    def __call__(self, x_test, y_test, var_test):
        x_baseline = x_test.copy()
        x_treatment = x_baseline.copy()
        non_categorical_vars = np.union1d(self.continuous_vars, self.discrete_vars)

        if var_test in non_categorical_vars:
            x_baseline[var_test] = self.agg[var_test](x_test[[var_test]], y_test)
        elif var_test in self.categorical_vars:
            dummy = self.categorical_vars[var_test]
            x_baseline[dummy] = self.agg[var_test](x_test[dummy], y_test)

        if not self.glob:
            for var in non_categorical_vars:
                if var != var_test:
                    x_baseline[var] = self.agg[var](x_test[[var]], y_test)
                    x_treatment[var] = x_baseline[var].copy()
            for var in self.categorical_vars:
                if var != var_test:
                    dummy = self.categorical_vars[var]
                    x_baseline[dummy] = self.agg[var](x_test[dummy], y_test)
                    x_treatment[dummy] = x_baseline[dummy].copy()

        return x_baseline, x_treatment
