import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class KNNConditionalSampler:
    """
    Simple k-NN based conditional sampler.

    For a target variable (possibly represented by multiple columns), this
    sampler re-draws its values by sampling from k-nearest neighbors in the
    training data, conditioning on a set of covariates.
    """

    def __init__(self, n_neighbors=10, conditioning_cols=None, random_state=None):
        self.n_neighbors = n_neighbors
        self.conditioning_cols = conditioning_cols
        self.random_state = np.random.RandomState(random_state) if random_state is not None else np.random
        self._fitted = False

    def fit(self, x_train: pd.DataFrame):
        if self.conditioning_cols is None:
            self.conditioning_cols_ = list(x_train.columns)
        else:
            self.conditioning_cols_ = [c for c in self.conditioning_cols if c in x_train.columns]

        if len(self.conditioning_cols_) == 0:
            raise ValueError("conditioning_cols resolved to an empty set of columns.")

        self._nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        self._nn.fit(x_train[self.conditioning_cols_].to_numpy())
        self._x_train = x_train
        self._fitted = True
        return self

    def __call__(self, x_train: pd.DataFrame, x_test: pd.DataFrame, cols):
        if not self._fitted:
            self.fit(x_train)

        cols = list(cols)
        X_cond_test = x_test[self.conditioning_cols_].to_numpy()
        _, indices = self._nn.kneighbors(X_cond_test, return_distance=True)

        rand_idx = self.random_state.randint(0, indices.shape[1], size=indices.shape[0])
        chosen = indices[np.arange(indices.shape[0]), rand_idx]

        baseline = self._x_train.iloc[chosen][cols].copy()
        baseline.index = x_test.index
        return baseline

