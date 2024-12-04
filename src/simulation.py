import numpy as np
from keras.utils import set_random_seed
from sklearn.model_selection import train_test_split
from scipy.special import expit

class True_Model:
    def predict(self, x, **args):
        return (3 + 4*x[:,1] + x[:,1]*x[:,2] + 3*x[:,3]**2 + 2*x[:,4]*x[:,5] + 6*x[:,6] + 2*np.sin(x[:,7]) + np.exp(x[:,8]) + 5*x[:,9] + 3*x[:,10] + 4*x[:,11] + 5*x[:,12])


def sim_reg_model(n_train, n_test, seed=0, split=True, sigma_epsilon=1):
    set_random_seed(seed)

    x = np.zeros((n_train + n_test, 20))

    # Intercept
    x[:, 0] = 1

    # Original: X1 - X7
    cov_matrix = np.diag(np.repeat(1.0, 7))
    cov_matrix[0, 5] = 0.85
    cov_matrix[5, 0] = 0.85

    x[:, 1:8] = np.random.multivariate_normal(np.zeros(7), cov_matrix, n_train + n_test)

    # Exponential: X8
    x[:, 8] = np.random.uniform(-1, 1, size=(n_train + n_test))

    # Categorical: X9
    z = np.random.normal(0, 1, size=(n_train + n_test))
    x[:, 9] = x[:, 2] + z < 0

    # Discrete: X10
    x[:, 10] = np.random.poisson(3, size=(n_train + n_test))

    # Fat-Tailed: X11, X12
    x[:, 11:13] = np.random.standard_t(5, size=(n_train + n_test, 2))

    # Noise
    epsilon = np.random.normal(0, sigma_epsilon, n_train + n_test)

    # Insignificant Variables: X13 - X19
    x[:, 13:15] = np.random.normal(0, 1, size=(n_train + n_test, 2))

    cov_matrix = np.diag(np.repeat(1.0, 2))
    cov_matrix[0, 1] = 0.85
    cov_matrix[1, 0] = 0.85
    x[:, 15:17] = np.random.multivariate_normal(np.zeros(2), cov_matrix, n_train + n_test)

    x[:, 17:20] = np.random.standard_t(5, size=(n_train + n_test, 3))

    # Y
    y = True_Model().predict(x) + epsilon
    
    if split:
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=n_train, random_state=seed)
        return x_train, x_test, y_train, y_test
    else:
        return x, y


def sim_class_model(n_train, n_test, seed=0):
    x, p = sim_reg_model(n_train, n_test, seed, split=False, sigma_epsilon=0)
    p = expit(-p)
    y = np.random.binomial(1, p)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=n_train, random_state=seed, stratify=y)
    return x_train, x_test, y_train, y_test
