import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from math import ceil
from sklearn.model_selection import train_test_split
from typing import Optional

from tf_model import CNN, compile_model
from config import config


from matplotlib import pyplot as plt
from model_validate import show_digit_predictions

def train(X_train: np.ndarray[any, np.dtype[np.float32]], y_train: np.ndarray[int], epochs: Optional[int] = 10, 
          validation_data: Optional[tuple[np.ndarray[any, np.dtype[np.float32]], np.ndarray[int]]] = None, 
          validation_split: Optional[float] = None, save_path: Optional[str] = 'model') -> None:
    """
    This function trains the model on the given data and saves the best model in the specified save path.
    Args:
    - X_train : numpy array, the training data.
    - y_train : numpy array, the labels of the training data.
    - epochs : int, number of epochs to train the model. Default is 10.
    - validation_data : tuple, data to use as validation data. Default is None.
    - validation_split : float, the proportion of data to use as validation data. Default is None.
    - save_path : str, the path to save the best model. Default is 'model'.
    """
    # if validation data is not provided, split the training data into training and validation datasets
    if (validation_data is None) and (validation_split is not None):
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_split, random_state=42)
        validation_data = (X_val, y_val)

    model = CNN(output_shape=config['num_classes'], scale=config['model']['filter_scale'], dropout=config['model']['dropout_rate'], 
                gaussian_noise=config['model']['gaussian_noise'], weight_decay=config['model']['weight_decay'],
                batch_momentum=config['model']['batch_momentum'])
    compile_model(model=model, lr=config['training']['learning_rate'])
    model_checkpoint = os.path.join(save_path, config['training']['model_checkpoint'])
    # stop the training if validation loss is not getting smaller
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5+ceil(epochs*0.2),
                                                  mode='auto', verbose=1),
                 # save the best model by validation loss during training
                tf.keras.callbacks.ModelCheckpoint(model_checkpoint, save_best_only=True, 
                                                  monitor='val_loss', mode='auto')]

    # Create an instance of the ImageDataGenerator class
    datagen = ImageDataGenerator(
        rotation_range=config['training']['augment']['rotation_range'],
        width_shift_range=config['training']['augment']['width_shift_range'],
        height_shift_range=config['training']['augment']['height_shift_range'],
        shear_range=config['training']['augment']['shear_range'],
        zoom_range=config['training']['augment']['zoom_range'],
        horizontal_flip=config['training']['augment']['horizontal_flip'],
        fill_mode=config['training']['augment']['fill_mode'])

    iterator = datagen.flow(X_train, y_train, batch_size=config['training']['batch_size'])
    # show_digit_predictions(batchX.reshape(20, 28, 28), batchY)

    print('Fit start')
    history = model.fit(iterator, callbacks=callbacks, verbose=1, steps_per_epoch=len(X_train) // config['training']['batch_size'], 
                        epochs=epochs, validation_data=(X_val, y_val))

    # model.save(model_path) # if last state of model is necessary
    # tf.keras.models.save_model(model, model_path)

def evaluate(X: np.ndarray[any, np.dtype[np.float32]], save_path: Optional[str] = 'model', 
             batch_size: Optional[int] = 1024) -> np.ndarray[any, np.dtype[int]]:
    """
    This function evaluates the model on the given data and returns the predictions.
    Args:
    - X : numpy array, the data to evaluate the model on.
    - save_path : str, the path where the model is saved. Default is 'model'.
    - batch_size : int, the batch size to use while evaluating the model. Default is 1024.
    Returns:
    - predictions : numpy array, the predictions of the model on the given data.
    """
    model_path = os.path.join(save_path, 'model.h5')
    # loading the model
    model = tf.keras.models.load_model(model_path)
    # predicting using the model
    predictions = model.predict(X, batch_size=batch_size)
    # choose the most probable class 
    predictions = np.argmax(predictions, axis=-1)
    return predictions
