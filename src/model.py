import os
import numpy as np
import pandas as pd
import keras
import keras_tuner
from sklearn.model_selection import train_test_split


def build_model(hp):
    # Specify fixed variable
    input_size = hp.Fixed('input_size', 76)

    # Define search space
    lr = hp.Float('lr', min_value=1e-8, max_value=10, step=10, sampling='log', default=0.01)
    l2_kernel = keras.regularizers.l2(hp.Float('l2_kernel', min_value=1e-8, max_value=10, step=10, sampling='log'))
    l2_bias = keras.regularizers.l2(hp.Float('l2_bias', min_value=1e-8, max_value=10, step=10, sampling='log'))
    n_layer_max = 4
    n_layer = hp.Int('n_layer', min_value=1, max_value=n_layer_max, default=2)
    units = []
    for i in range(n_layer):
        with hp.conditional_scope('n_layer', list(range(i+1, n_layer_max+1))):
            units.append(hp.Int(f'units_{i}', min_value=16, max_value=2048, step=2, sampling='log', default=1024))
    activation = hp.Choice('activation', ['relu', 'sigmoid'], default='sigmoid')
    dropout = hp.Float('dropout', min_value=0, max_value=0.9, step=0.1, default=0.5)

    # Construct model
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_size,)))
    for i in range(n_layer):
        model.add(keras.layers.Dense(units[i], activation=activation, kernel_regularizer=l2_kernel, bias_regularizer=l2_bias))
    model.add(keras.layers.Dropout(rate=dropout))
    model.add(keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2_kernel, bias_regularizer=l2_bias))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=[keras.losses.BinaryCrossentropy(), keras.metrics.AUC()])
    
    return model


def train_model(x_train, x_val, y_train, y_val, tuner_dir, model_dir, seed, hyperparameters=None):
    class_weight = class_weight = {0: len(y_train) / (2 * (y_train == 0).sum()),
                                   1: len(y_train) / (2 * (y_train == 1).sum())}

    tuner = keras_tuner.Hyperband(hypermodel=build_model,
                                  objective='val_auc',#'val_binary_crossentropy',
                                  max_epochs=100,
                                  hyperband_iterations=3,
                                  overwrite=False,
                                  seed=seed,
                                  hyperparameters=hyperparameters,
                                  tune_new_entries=hyperparameters is None,
                                  directory=tuner_dir,
                                  project_name=str(seed))
    tuner.search(x_train, y_train,
                 epochs=100,
                 validation_data=(x_val, y_val),
                 class_weight=class_weight,
                 batch_size=32,
                 callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          min_delta=1e-5,
                                                          patience=50,
                                                          restore_best_weights=True),
                             keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                               factor=0.1,
                                                               min_delta=1e-5,
                                                               patience=15,
                                                               min_lr=1e-9)])
    best_model = tuner.get_best_models()[0]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    best_model.save(os.path.join(model_dir, f'{seed}.keras'))

def evaluate_model(x_train, x_val, x_test, y_train, y_val, y_test, model, metrics, pred_params=dict()):
    evals = []

    x = dict(train=x_train, val=x_val, test=x_test)
    y = dict(train=y_train, val=y_val, test=y_test)
    f = {data: np.array(model.predict(x[data], **pred_params), dtype=np.float64).flatten() for data in ['train', 'val', 'test']}

    # Evaluate each metric from the dictionary
    for metric_name, metric_func in metrics.items():
        for data in ['train', 'val', 'test']:
            metric = metric_func()
            evals.append(dict(data=data, metric=metric_name, result=float(metric(y[data], f[data]))))

    # Convert to DataFrame and pivot the results
    evals = pd.DataFrame(evals)
    evals = (evals
             .pivot(index='metric', columns='data', values='result')
             .sort_index(axis=1)
             [['train', 'val', 'test']])
    
    return evals

