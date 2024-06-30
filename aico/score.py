import numpy as np
import tensorflow as tf

def neg_squared_loss(predicted_y, true_y):
    return -np.square(true_y - np.squeeze(predicted_y))

def neg_absolute_loss(predicted_y, true_y):
    return -np.abs(true_y - np.squeeze(predicted_y))

def neg_binary_cross_entropy_loss(predicted_y, true_y):
    epsilon = tf.keras.backend.epsilon()
    predicted_y = np.clip(predicted_y, epsilon, 1 - epsilon)
    return np.log(np.squeeze(predicted_y))*true_y + np.log(1 - np.squeeze(predicted_y))*(1 - true_y)
