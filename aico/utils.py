import numpy as np
import pandas as pd
from functools import partial

def get_array_intercept(intercept_array):
    '''Helper function for preprocess_intercept. Construct the intercept from array
    '''
    def array_intercept(x, k, intercept_array):
        intercept = np.zeros((x.shape[0], intercept_array.shape[1]))
        intercept[:, :] = intercept_array[k, :]
        return intercept
    
    return partial(array_intercept, intercept_array=intercept_array)


def get_flipping_intercept(intercept_vector, flipping_var=[], round_flipping_var=True):
    '''Helper function to construct the flipped intercept

    Args:
        intercept_vector (numpy.ndarray): the intercept vector for non-flipped variables
        flipping_var (list): the list of variables to which the flipped intercept method will be applied.
        round_flipping_var (boolean): indicate if the variables in flipping_var not currently tested will 
                                      be rounded to 0 or 1 (to avoid out-of-distribution model evaluation)
    '''
    def flipping_intercept(x, k, j=None, intercept_vector=None, flipping_var=[], round_flipping_var=True):
        intercept = np.zeros((x.shape[0], intercept_vector.shape[0]))
        if round_flipping_var:
            intercept_vector[flipping_var] = np.round(intercept_vector[flipping_var])
        intercept[:, :] = intercept_vector
        if k in flipping_var:
            intercept[:, k] = 1 - x[:, k]
        if j in flipping_var:
            intercept[:, j] = 1 - x[:, j]
        return intercept
    
    return partial(flipping_intercept,
                   intercept_vector=intercept_vector,
                   flipping_var=flipping_var,
                   round_flipping_var=round_flipping_var)
    

def preprocess_intercept(intercept):
    '''Proprocess the intercepts
    '''
    if callable(intercept):
        pass
    elif isinstance(intercept, np.ndarray) or isinstance(intercept, pd.DataFrame):
        intercept = np.array(intercept)
        if intercept.ndim == 1:
            intercept_array = np.zeros((intercept.shape[0], intercept.shape[0]))
            intercept_array[:, :] = intercept
        else:
            intercept_array = intercept
        intercept = get_array_intercept(intercept_array)
    else:
        raise Exception("Intercept must be callable, NumPy Array, or Pandas DataFrame")
    return intercept


def preprocess_xy(x, y=None):
    '''Proprocess the data
    '''
    if isinstance(x, np.ndarray):
        var_name = [f'X{k}' for k in range(x.shape[1])]
    elif isinstance(x, pd.DataFrame):
        var_name = list(x.columns)
        x = np.array(x)
    else:
        raise Exception('x must be either Numpy array or Pandas dataframe')
    
    if y is None:
        return x, var_name
    else:
        return x, var_name, np.array(y)