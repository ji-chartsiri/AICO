class MeanAgg:
    def __call__(self, x_train, y_train, pred_func):
        """
        Compute the mean aggregation for a continuous variable.

        Parameters:
        - x_train (pd.DataFrame): Feature data used for training.
        - y_train (pd.Series or np.array): Response data used for training.
        - pred_func (callable): Prediction function of the model.

        Returns:
        - agg (callable): Aggregation function for the mean value.
        """
        x_baseline = x_train.mean().iloc[0]
        def agg(x_test, y_test):
            return x_baseline
        return agg
    
class QuantileAgg:
    def __init__(self, quantile):
        """
        Initialize the quantile aggregation with a specific quantile.

        Parameters:
        - quantile (float): Quantile value to use for aggregation (e.g., 0.5 for median).
        """
        self.quantile = quantile

    def __call__(self, x_train, y_train, pred_func):
        """
        Compute the quantile aggregation for a continuous variable.

        Parameters:
        - x_train (pd.DataFrame): Feature data used for training.
        - y_train (pd.Series or np.array): Response data used for training.
        - pred_func (callable): Prediction function of the model.

        Returns:
        - agg (callable): Aggregation function for the quantile value.
        """
        x_baseline = x_train.quantile(self.quantile).iloc[0]
        def agg(x_test, y_test):
            return x_baseline
        return agg

class MedianAgg(QuantileAgg):
    def __init__(self):
        """
        Initialize the median aggregation as a special case of the quantile aggregation.
        """
        super().__init__(quantile=0.5)

class ModeAgg:
    def __call__(self, x_train, y_train, pred_func):
        """
        Compute the mode aggregation for a discrete or categorical variable.

        Parameters:
        - x_train (pd.DataFrame): Feature data used for training.
        - y_train (pd.Series or np.array): Response data used for training.
        - pred_func (callable): Prediction function of the model.

        Returns:
        - agg (callable): Aggregation function for the mode value.
        """
        x_baseline = (x_train
                      .value_counts(sort=True, ascending=False)
                      .reset_index(name='count')
                      .drop(columns='count')
                      .iloc[0])
        def agg(x_test, y_test):
            return x_baseline
        return agg

class AltModeAgg:
    def __call__(self, x_train, y_train, pred_func):
        """
        Compute an alternative mode aggregation that considers the two most frequent values.

        Parameters:
        - x_train (pd.DataFrame): Feature data used for training.
        - y_train (pd.Series or np.array): Response data used for training.
        - pred_func (callable): Prediction function of the model.

        Returns:
        - agg (callable): Aggregation function that returns the mode value or the second mode if needed.
        """
        x_mode = (x_train
                  .value_counts(sort=True, ascending=False)
                  .reset_index(name='count')
                  .drop(columns='count')
                  .iloc[:2])
        def agg(x_test, y_test):
            x_baseline = x_test.copy()
            x_baseline.loc[:, :] = x_mode.iloc[0].values
            x_baseline.loc[x_test.eq(x_mode.iloc[0]).all(axis=1), :] = x_mode.iloc[1].values
            return x_baseline
        return agg
