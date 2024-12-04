import numpy as np

def neg_squared_loss(predicted_y, true_y):
    """
    Compute the negative squared loss between the predicted and true values.

    Parameters:
    - predicted_y (np.array or pd.Series): Predicted values from the model.
    - true_y (np.array or pd.Series): True values.

    Returns:
    - loss (np.array): Negative squared loss for each observation.
    """
    return -np.square(true_y - np.squeeze(predicted_y))

def neg_absolute_loss(predicted_y, true_y):
    """
    Compute the negative absolute loss between the predicted and true values.

    Parameters:
    - predicted_y (np.array or pd.Series): Predicted values from the model.
    - true_y (np.array or pd.Series): True values.

    Returns:
    - loss (np.array): Negative absolute loss for each observation.
    """
    return -np.abs(true_y - np.squeeze(predicted_y))

def neg_binary_cross_entropy_loss(predicted_y, true_y):
    """
    Compute the negative binary cross-entropy loss between the predicted and true values.

    Parameters:
    - predicted_y (np.array or pd.Series): Predicted probabilities from the model.
    - true_y (np.array or pd.Series): True binary labels (0 or 1).

    Returns:
    - loss (np.array): Negative binary cross-entropy loss for each observation.
    """
    predicted_y = (np.array(predicted_y)
                   .flatten()
                   .astype(np.longdouble))

    predicted_y_0 = np.clip(-predicted_y[true_y == 0],
                            np.nextafter(-1, 0, dtype=np.longdouble),
                            None)

    predicted_y_1 = np.clip(predicted_y[true_y == 1],
                            np.nextafter(0, 1, dtype=np.longdouble),
                            None)

    loss = np.zeros_like(predicted_y, dtype=np.longdouble)
    loss[true_y == 0] = np.log1p(predicted_y_0)
    loss[true_y == 1] = np.log(predicted_y_1)
    return loss

def neg_logit_loss(predicted_y, true_y):
    """
    Compute the negative logit loss between the predicted and true values.

    Parameters:
    - predicted_y (np.array or pd.Series): Predicted probabilities from the model.
    - true_y (np.array or pd.Series): True binary labels (0 or 1).

    Returns:
    - loss (np.array): Negative logit loss for each observation.
    """
    predicted_y = (predicted_y
                   .flatten()
                   .astype(np.longdouble))
    
    predicted_y = np.clip(predicted_y,
                          np.nextafter(0, 1, dtype=np.longdouble),
                          np.nextafter(1, 0, dtype=np.longdouble))

    loss = np.zeros_like(predicted_y, dtype=np.longdouble)
    loss[true_y == 0] = np.log1p(-predicted_y[true_y == 0]) - np.log(predicted_y[true_y == 0])
    loss[true_y == 1] = np.log(predicted_y[true_y == 1]) - np.log1p(-predicted_y[true_y == 1])

    return loss
