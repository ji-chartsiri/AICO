from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1, l2
# from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
import tensorflow as tf
# tf.keras.backend.set_floatx('float64')


def build_reg_model(input_size, n_hidden_layer, n_per_hidden_layer=128,
                    hidden_activation='relu', model_loss='mean_squared_error',
                    regularize=0, regularize_weight=1e-5, learning_rate=0.001):
    if regularize == 0:
        regularize = None
    elif regularize == 1:
        regularize = l1(regularize_weight)
    elif regularize == 2:
        regularize = l2(regularize_weight)

    
    inputs = Input(shape=(input_size,))

    prev_layer = Dense(n_per_hidden_layer, activation=hidden_activation, kernel_regularizer=regularize)(inputs)
    for _ in range(n_hidden_layer - 1):
        prev_layer = Dense(n_per_hidden_layer, activation=hidden_activation, kernel_regularizer=regularize)(prev_layer)

    output = Dense(1, activation='linear')(prev_layer)
    model = Model(inputs=inputs,
                  outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=model_loss)
    return model

def train_reg_model(x_train, y_train, n_hidden_layer, n_per_hidden_layer=128, hidden_activation='relu', 
                    model_loss='mean_squared_error', validation_split=0.25, batch_size=32, nr_epochs=50,
                    min_delta=1e-5, patience=5, regularize=0, regularize_weight=1e-5, learning_rate=0.001):

    model = build_reg_model(x_train.shape[1], n_hidden_layer, n_per_hidden_layer,
                            hidden_activation, model_loss, regularize, regularize_weight, learning_rate)
    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=min_delta,
                               patience=patience,
                               restore_best_weights=True)
    model.fit(x=x_train,
              y=y_train,
              batch_size=batch_size,
              epochs=nr_epochs,
              validation_split=validation_split,
              callbacks=[early_stop],
              verbose=0)

    return model

def train_class_model(x_train, y_train, n_hidden_layer, n_per_hidden_layer=128,
                      hidden_activation='relu', model_loss='binary_crossentropy',
                      validation_split=0.25, learning_rate=0.001):
    batch_size = 32
    nr_epochs = 150
    inputs = Input(shape=(x_train.shape[1],))
    min_delta = 1e-5

    prev_layer = Dense(n_per_hidden_layer, activation=hidden_activation)(inputs)
    for _ in range(n_hidden_layer - 1):
        prev_layer = Dense(n_per_hidden_layer, activation=hidden_activation)(prev_layer)
    output = Dense(1, activation='sigmoid')(prev_layer)

    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=min_delta,
                               patience=5)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=model_loss)

    model.fit(x=x_train,
              y=y_train,
              batch_size=batch_size,
              epochs=nr_epochs,
              validation_split=validation_split,
              callbacks=[early_stop],
              verbose=0)
    # tf.keras.backend.clear_session()
    
    return model