"""
This script is responisble the different model architectures.
"""

# Import libraries
#import numpy as np, matplotlib.pyplot as plt, seaborn as sns, keras_tuner as kt,
import tensorflow as tf
import sys #os, datetime, progressbar
import tensorflow_probability as tfp

tfd = tfp.distributions

#from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, Flatten, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
#from tensorflow.keras.datasets import mnist
#from tensorflow.keras.utils import to_categorical, plot_model
#from keras.callbacks import ModelCheckpoint#, TensorBoard
from keras import backend as K
#from livelossplot import PlotLossesKeras

from keras_tuner.tuners import RandomSearch
#from keras_tuner.engine.hypermodel import HyperModel
from keras_tuner.engine.hyperparameters import HyperParameters
#from tensorboard.plugins.hparams import api as hp
#from keras.utils.generic_utils import get_custom_objects

import gc
gc.enable()

#color = 30

# import module
sys.path.insert(1, '/Users/tashaleebillings/Desktop/Tasha_Desktop/ThesisResearch/ML/ML_Paper2_Scripts/')
import utilities as utils


############################################################
# Models trained using probabilistic loss functions
############################################################

def dropout_model1_dloss(X_train, y_train, X_test, y_test, nx, ny, color, epochs, batch_size, loss_name, model_name, path, filter, trainig_data_name, training = True):
    # Making a model.
    """
    Parameters
    ----------
    KL_DIVERGENCE_FUNCTION : str
        Divergence function.
    X_TRAIN : ndarray
        Input image data before the CNN is trained.
    Y_TRAIN : ndarray
        Image labels for the CNN while it is being trained.
    X_TEST : ndarray
        Input image data after the CNN is trained it is tested with this dataset.
    Y_TEST : ndarray
        Image labels. After the CNN is trained it is tested with this dataset.
    EPOCH : int
        Number of training steps. The number of times to go through the full training data.
    BATCH_SIZE : int
        Number of training data required to update the model.
    LOSS_NAME : str
        The prob. loss function used duing the training process. It's probably a custom loss funtion.
        eg. "elbo" or "nll"
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    TRAINING : bool
        False means dropout is not activated during dropout
    """
    input0 = Input(shape=(nx, ny, color))
    inner = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu')(input0)
    inner = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu')(inner)
    
    inner = Flatten()(inner)
    inner = Dropout(0.2)(inner, training=training)
    output = Dense(1, activation='linear')(inner)
    
    model = Model(inputs=input0, outputs=output)

    model.compile(optimizer= "adam", loss = loss_name, metrics=["mse"])
    
    #plotlosses = PlotLossesKeras()

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=epochs, batch_size=batch_size,
              #callbacks=[plotlosses],
              verbose=1)
    print("Model Training complete... ")

    # Visualize Model
    print(model.summary())

    loss = model.evaluate(X_test, y_test)
    
    # save model and model_weights
    print("Saving model and model weights ...")
    utils.save_model_modelweights(bcnn_model=model, path=path, filter=filter, trainig_data_name=trainig_data_name, model_name=model_name, loss_name=loss_name)
    
    return model#, history

def dropout_model2_dloss(X_train, y_train, X_test, y_test, nx, ny, color, epochs, batch_size, loss_name, model_name, path, filter, trainig_data_name, training = True):
    # Making a model.
    """
    Parameters
    ----------
    KL_DIVERGENCE_FUNCTION : str
        Divergence function.
    X_TRAIN : ndarray
        Input image data before the CNN is trained.
    Y_TRAIN : ndarray
        Image labels for the CNN while it is being trained.
    X_TEST : ndarray
        Input image data after the CNN is trained it is tested with this dataset.
    Y_TEST : ndarray
        Image labels. After the CNN is trained it is tested with this dataset.
    EPOCH : int
        Number of training steps. The number of times to go through the full training data.
    BATCH_SIZE : int
        Number of training data required to update the model.
    LOSS_NAME : str
        The prob. loss function used duing the training process. It's probably a custom loss funtion.
        eg. "elbo" or "nll"
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    TRAINING : bool
        False means dropout is not activated during dropout
    """

    input0 = Input(shape=(nx, ny, color))
    inner = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding = 'valid')(input0)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    
    inner = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding = 'same')(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    
    inner = Flatten()(inner)
    inner = Dropout(0.2)(inner, training=training)
    output = Dense(1, activation='linear')(inner)
    
    model = Model(inputs=input0, outputs=output)

    model.compile(optimizer= "adam", loss = loss_name, metrics=["mse"])
    
    #plotlosses = PlotLossesKeras()

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=epochs, batch_size=batch_size,
              #callbacks=[plotlosses],
              verbose=1)
    print("Model Training complete... ")

    # Visualize Model
    print(model.summary())

    loss = model.evaluate(X_test, y_test)
    
    # save model and model_weights
    print("Saving model and model weights ...")
    utils.save_model_modelweights(bcnn_model=model, path=path, filter=filter, trainig_data_name=trainig_data_name, model_name=model_name, loss_name=loss_name)
    
    return model#, history

def dropout_model3_dloss(X_train, y_train, X_test, y_test, nx, ny, color, epochs, batch_size, loss_name, model_name, path, filter, trainig_data_name, training = True):
    # Making a model.
    """
    Parameters
    ----------
    KL_DIVERGENCE_FUNCTION : str
        Divergence function.
    X_TRAIN : ndarray
        Input image data before the CNN is trained.
    Y_TRAIN : ndarray
        Image labels for the CNN while it is being trained.
    X_TEST : ndarray
        Input image data after the CNN is trained it is tested with this dataset.
    Y_TEST : ndarray
        Image labels. After the CNN is trained it is tested with this dataset.
    EPOCH : int
        Number of training steps. The number of times to go through the full training data.
    BATCH_SIZE : int
        Number of training data required to update the model.
    LOSS_NAME : str
        The prob. loss function used duing the training process. It's probably a custom loss funtion.
        eg. "elbo" or "nll"
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    TRAINING : bool
        False means dropout is not activated during dropout
    """

    input0 = Input(shape=(nx, ny, color))
    inner = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding = 'valid')(input0)
    inner = BatchNormalization()(inner)

    inner = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding = 'valid')(inner)
    inner = BatchNormalization()(inner)

    inner = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding = 'valid')(inner)
    inner = BatchNormalization()(inner)

    inner = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding = 'valid')(inner)
    inner = BatchNormalization()(inner)

    inner = Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding = 'valid')(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    
    inner = Flatten()(inner)
    inner = Dropout(0.2)(inner, training=training)
    output = Dense(1, activation='linear')(inner)
    
    model = Model(inputs=input0, outputs=output)

    model.compile(optimizer= "adam", loss = loss_name, metrics=["mse"])

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=epochs, batch_size=batch_size,
              #callbacks=[plotlosses],
              verbose=1)

    # Visualize Model
    print(model.summary())

    loss = model.evaluate(X_test, y_test)
    
    # save model and model_weights
    print("Saving model and model weights ...")
    utils.save_model_modelweights(bcnn_model=model, path=path, filter=filter, trainig_data_name=trainig_data_name, model_name=model_name, loss_name=loss_name)
    
    return model#, history

def dropout_model4_dloss(X_train, y_train, X_test, y_test, nx, ny, color, epochs, batch_size, loss_name, model_name, path, filter, trainig_data_name, training = True):
    # Making a model.
    """
    Parameters
    ----------
    KL_DIVERGENCE_FUNCTION : str
        Divergence function.
    X_TRAIN : ndarray
        Input image data before the CNN is trained.
    Y_TRAIN : ndarray
        Image labels for the CNN while it is being trained.
    X_TEST : ndarray
        Input image data after the CNN is trained it is tested with this dataset.
    Y_TEST : ndarray
        Image labels. After the CNN is trained it is tested with this dataset.
    EPOCH : int
        Number of training steps. The number of times to go through the full training data.
    BATCH_SIZE : int
        Number of training data required to update the model.
    LOSS_NAME : str
        The prob. loss function used duing the training process. It's probably a custom loss funtion.
        eg. "elbo" or "nll"
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    TRAINING : bool
        False means dropout is not activated during dropout
    """

    input0 = Input(shape=(nx, ny, color))
    inner = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding = 'valid')(input0)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)

    inner = Conv2D(filters=28, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding = 'valid')(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    
    inner = Flatten()(inner)
    
    inner = Dropout(0.2)(inner, training=training)
    inner = Dense(100, activation='linear')(inner)
    
    inner = Dropout(0.2)(inner, training=training)
    inner = Dense(10, activation='linear')(inner)
    
    inner = Dropout(0.2)(inner, training=training)
    output = Dense(1, activation='linear')(inner)
    
    model = Model(inputs=input0, outputs=output)

    model.compile(optimizer= "adam", loss = loss_name, metrics=["mse"])

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=epochs, batch_size=batch_size,
              #callbacks=[plotlosses],
              verbose=1)
              
    # Visualize Model
    print(model.summary())

    loss = model.evaluate(X_test, y_test)
    
    # save model and model_weights
    print("Saving model and model weights ...")
    utils.save_model_modelweights(bcnn_model=model, path=path, filter=filter, trainig_data_name=trainig_data_name, model_name=model_name, loss_name=loss_name)
    
    return model#, history




############################################################
# Models trained using probabilistic loss functions
############################################################

def model1_ploss(X_train, y_train, X_test, y_test, nx, ny, color, epochs, batch_size, loss_name, model_name, path, filter, trainig_data_name,  kl_divergence_function):
    # Making a model.
    """
    Parameters
    ----------
    KL_DIVERGENCE_FUNCTION : str
        Divergence function.
    X_TRAIN : ndarray
        Input image data before the CNN is trained.
    Y_TRAIN : ndarray
        Image labels for the CNN while it is being trained.
    X_TEST : ndarray
        Input image data after the CNN is trained it is tested with this dataset.
    Y_TEST : ndarray
        Image labels. After the CNN is trained it is tested with this dataset.
    EPOCH : int
        Number of training steps. The number of times to go through the full training data.
    BATCH_SIZE : int
        Number of training data required to update the model.
    LOSS_NAME : str
        The prob. loss function used duing the training process. It's probably a custom loss funtion.
        eg. "elbo" or "nll"
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    """
    model = Sequential()
    
    model.add(tfp.layers.Convolution2DFlipout(filters = 16,
                                              input_shape=(nx, ny, color), kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              #, padding = 'same'
                                             ))
    model.add(tfp.layers.Convolution2DFlipout(filters = 8,
                                              kernel_size=3, activation = 'relu',
                                              kernel_divergence_fn=kl_divergence_function,
                                              #padding = 'same'
                                             ))
    model.add(Flatten())
    model.add(tfp.layers.DenseFlipout(1, activation="linear", kernel_divergence_fn=kl_divergence_function,))
    model.add(tfp.layers.DistributionLambda(
      lambda t: tfd.Normal(loc=t[..., :1],
                           #scale = 1)))
                           scale=1e-3 + tf.math.softplus(0.01 * t[..., :1]))))
    

    model.compile(optimizer= "adam", loss = loss_name, metrics=["mse"])
    
    #plotlosses = PlotLossesKeras()

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
            epochs=epochs, batch_size=batch_size,
            #callbacks=[plotlosses],
            verbose=1)
    print("Model Training complete... ")

    # Visualize Model
    print(model.summary())

    loss = model.evaluate(X_test, y_test)
    
    # save model and model_weights
    print("Saving model and model weights ...")
    utils.save_model_modelweights(bcnn_model=model, path=path, filter=filter, trainig_data_name=trainig_data_name, model_name=model_name, loss_name=loss_name)
    
    return model#, history
    
def model2_ploss(X_train, y_train, X_test, y_test, nx, ny, color, epochs, batch_size, loss_name, model_name, path, filter, trainig_data_name,  kl_divergence_function):

    # Making a model.
    """
    Parameters
    ----------
    KL_DIVERGENCE_FUNCTION : str
        Divergence function.
    X_TRAIN : ndarray
        Input image data before the CNN is trained.
    Y_TRAIN : ndarray
        Image labels for the CNN while it is being trained.
    X_TEST : ndarray
        Input image data after the CNN is trained it is tested with this dataset.
    Y_TEST : ndarray
        Image labels. After the CNN is trained it is tested with this dataset.
    EPOCH : int
        Number of training steps. The number of times to go through the full training data.
    BATCH_SIZE : int
        Number of training data required to update the model.
    LOSS_NAME : str
        The prob. loss function used duing the training process. It's probably a custom loss funtion.
        eg. "elbo" or "nll"
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    """
    
    model = Sequential()

    # Added Layers.
    model.add(tfp.layers.Convolution2DFlipout(filters = 16,
                                              input_shape=(nx, ny, color), kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , padding = 'valid',
                                              ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(tfp.layers.Convolution2DFlipout(filters = 8,
                                              kernel_size=3, activation = 'relu',
                                              kernel_divergence_fn=kl_divergence_function,
                                              padding = 'same'
                                              ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(tfp.layers.DenseFlipout(1, activation="linear", kernel_divergence_fn=kl_divergence_function,))
    model.add(tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                             #scale = 1)))
                             scale=1e-3 + tf.math.softplus(0.01 * t[..., :1]))))

    model.compile(optimizer= "adam", loss = loss_name, metrics=["mse"])

    #plotlosses = PlotLossesKeras()

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=epochs, batch_size=batch_size,
              #callbacks=[plotlosses],
              verbose=1)
    print("Model Training complete... ")

    # Visualize Model
    print(model.summary())

    loss = model.evaluate(X_test, y_test)
    
    # save model and model_weights
    print("Saving model and model weights ...")
    utils.save_model_modelweights(bcnn_model=model, path=path, filter=filter, trainig_data_name=trainig_data_name, model_name=model_name, loss_name=loss_name)
    
    return model#, history
    
def model3_ploss(X_train, y_train, X_test, y_test, nx, ny, color, epochs, batch_size, loss_name, model_name, path, filter, trainig_data_name,  kl_divergence_function):
    # Making a model.
    """
    Parameters
    ----------
    KL_DIVERGENCE_FUNCTION : str
        Divergence function.
    X_TRAIN : ndarray
        Input image data before the CNN is trained.
    Y_TRAIN : ndarray
        Image labels for the CNN while it is being trained.
    X_TEST : ndarray
        Input image data after the CNN is trained it is tested with this dataset.
    Y_TEST : ndarray
        Image labels. After the CNN is trained it is tested with this dataset.
    EPOCH : int
        Number of training steps. The number of times to go through the full training data.
    BATCH_SIZE : int
        Number of training data required to update the model.
    LOSS_NAME : str
        The prob. loss function used duing the training process. It's probably a custom loss funtion.
        eg. "elbo" or "nll"
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    """
    model = Sequential()

    # Added Layers.
    model.add(tfp.layers.Convolution2DFlipout(filters = 32,
                                              input_shape=(nx, ny, color), kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , padding = 'valid',
                                              ))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(tfp.layers.Convolution2DFlipout(filters = 32, kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , padding = 'valid',
                                              ))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))
                          
    model.add(tfp.layers.Convolution2DFlipout(filters = 16, kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , padding = 'valid',
                                              ))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(tfp.layers.Convolution2DFlipout(filters = 8, kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , padding = 'valid',
                                              ))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(tfp.layers.Convolution2DFlipout(filters = 4, kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , padding = 'valid',
                                              ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(tfp.layers.DenseFlipout(1, activation="linear", kernel_divergence_fn=kl_divergence_function,))
    model.add(tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                             #scale = 1)))
                             scale=1e-3 + tf.math.softplus(0.01 * t[..., :1]))))

    model.compile(optimizer= "adam", loss = loss_name, metrics=["mse"])

    #plotlosses = PlotLossesKeras()

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=epochs, batch_size=batch_size,
              #callbacks=[plotlosses],
              verbose=1)
    print("Model Training complete... ")

    # Visualize Model
    print(model.summary())

    loss = model.evaluate(X_test, y_test)
    
    # save model and model_weights
    print("Saving model and model weights ...")
    utils.save_model_modelweights(bcnn_model=model, path=path, filter=filter, trainig_data_name=trainig_data_name, model_name=model_name, loss_name=loss_name)
    
    return model#, history
    
def model4_ploss(X_train, y_train, X_test, y_test, nx, ny, color, epochs, batch_size, loss_name, model_name, path, filter, trainig_data_name,  kl_divergence_function):
    # Making a model.
    """
    Parameters
    ----------
    KL_DIVERGENCE_FUNCTION : str
        Divergence function.
    X_TRAIN : ndarray
        Input image data before the CNN is trained.
    Y_TRAIN : ndarray
        Image labels for the CNN while it is being trained.
    X_TEST : ndarray
        Input image data after the CNN is trained it is tested with this dataset.
    Y_TEST : ndarray
        Image labels. After the CNN is trained it is tested with this dataset.
    EPOCH : int
        Number of training steps. The number of times to go through the full training data.
    BATCH_SIZE : int
        Number of training data required to update the model.
    LOSS_NAME : str
        The prob. loss function used duing the training process. It's probably a custom loss funtion.
        eg. "elbo" or "nll"
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    """
    model = Sequential()

    # Added Layers.
    model.add(tfp.layers.Convolution2DFlipout(filters = 32,
                                              input_shape=(nx, ny, color), kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , padding = 'valid',
                                              ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(tfp.layers.Convolution2DFlipout(filters = 28, kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , padding = 'valid',
                                              ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dropout(0.2))
    model.add(tfp.layers.DenseFlipout(100, activation="linear", kernel_divergence_fn=kl_divergence_function,))
    
    model.add(Dropout(0.2))
    model.add(tfp.layers.DenseFlipout(10, activation="linear", kernel_divergence_fn=kl_divergence_function,))

    model.add(Dropout(0.2))
    model.add(tfp.layers.DenseFlipout(1, activation="linear", kernel_divergence_fn=kl_divergence_function,))
    model.add(tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1],
                             #scale = 1)))
                             scale=1e-3 + tf.math.softplus(0.01 * t[..., :1]))))

    model.compile(optimizer= "adam", loss = loss_name, metrics=["mse"])

    #plotlosses = PlotLossesKeras()

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=epochs, batch_size=batch_size,
              #callbacks=[plotlosses],
              verbose=1)
    print("Model Training complete... ")

    # Visualize Model
    print(model.summary())

    loss = model.evaluate(X_test, y_test)
    
    # save model and model_weights
    print("Saving model and model weights ...")
    utils.save_model_modelweights(bcnn_model=model, path=path, filter=filter, trainig_data_name=trainig_data_name, model_name=model_name, loss_name=loss_name)
    
    return model#, history


############################################################
# Models trained using deterministic loss functions
############################################################
    

def model1_dloss(X_train, y_train, X_test, y_test, nx, ny, color, epochs, batch_size, loss_name, model_name, path, filter, trainig_data_name,  kl_divergence_function):
    # Making a model.
    """
    Parameters
    ----------
    KL_DIVERGENCE_FUNCTION : str
        Divergence function.
    X_TRAIN : ndarray
        Input image data before the CNN is trained.
    Y_TRAIN : ndarray
        Image labels for the CNN while it is being trained.
    X_TEST : ndarray
        Input image data after the CNN is trained it is tested with this dataset.
    Y_TEST : ndarray
        Image labels. After the CNN is trained it is tested with this dataset.
    EPOCH : int
        Number of training steps. The number of times to go through the full training data.
    BATCH_SIZE : int
        Number of training data required to update the model.
    LOSS_NAME : str
        The prob. loss function used duing the training process. It's probably a custom loss funtion.
        eg. "elbo" or "nll"
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    """
    model = Sequential()
    
    model.add(tfp.layers.Convolution2DFlipout(filters = 16,
                                              input_shape=(nx, ny, color), kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              #, padding = 'same'
                                             ))
    model.add(tfp.layers.Convolution2DFlipout(filters = 8,
                                              kernel_size=3, activation = 'relu',
                                              kernel_divergence_fn=kl_divergence_function,
                                              #padding = 'same'
                                             ))
    model.add(Flatten())
    model.add(tfp.layers.DenseFlipout(1, activation="linear", kernel_divergence_fn=kl_divergence_function,))
    
    #for l in range(hp.Int('loss', 0, len(losses))):
    model.compile(optimizer= "adam", loss = loss_name, metrics=["mse"])
    
    #plotlosses = PlotLossesKeras()

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=epochs, batch_size=batch_size,
              #callbacks=[plotlosses],
              verbose=1)
    print("Model Training complete... ")

    # Visualize Model
    print(model.summary())

    loss = model.evaluate(X_test, y_test)
    
    # save model and model_weights
    print("Saving model and model weights ...")
    utils.save_model_modelweights(bcnn_model=model, path=path, filter=filter, trainig_data_name=trainig_data_name, model_name=model_name, loss_name=loss_name)
    
    return model#, history

def model2_dloss(X_train, y_train, X_test, y_test, nx, ny, color, epochs, batch_size, loss_name, model_name, path, filter, trainig_data_name,  kl_divergence_function):

    # Making a model.
    """
    Parameters
    ----------
    KL_DIVERGENCE_FUNCTION : str
        Divergence function.
    X_TRAIN : ndarray
        Input image data before the CNN is trained.
    Y_TRAIN : ndarray
        Image labels for the CNN while it is being trained.
    X_TEST : ndarray
        Input image data after the CNN is trained it is tested with this dataset.
    Y_TEST : ndarray
        Image labels. After the CNN is trained it is tested with this dataset.
    EPOCH : int
        Number of training steps. The number of times to go through the full training data.
    BATCH_SIZE : int
        Number of training data required to update the model.
    LOSS_NAME : str
        The prob. loss function used duing the training process. It's probably a custom loss funtion.
        eg. "elbo" or "nll"
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    """
    model = Sequential()

    # Added Layers.
    model.add(tfp.layers.Convolution2DFlipout(filters = 16,
                                              input_shape=(nx, ny, color), kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , padding = 'valid',
                                              ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(tfp.layers.Convolution2DFlipout(filters = 8,
                                              kernel_size=3, activation = 'relu',
                                              kernel_divergence_fn=kl_divergence_function,
                                              padding = 'same'
                                              ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(tfp.layers.DenseFlipout(1, activation="linear", kernel_divergence_fn=kl_divergence_function,))

    #for l in range(hp.Int('loss', 0, len(losses))):
    model.compile(optimizer= "adam", loss = loss_name, metrics=["mse"])

    #plotlosses = PlotLossesKeras()

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=epochs, batch_size=batch_size,
              #callbacks=[plotlosses],
              verbose=1)
    print("Model Training complete... ")

    # Visualize Model
    print(model.summary())

    loss = model.evaluate(X_test, y_test)
    
    # save model and model_weights
    print("Saving model and model weights ...")
    utils.save_model_modelweights(bcnn_model=model, path=path, filter=filter, trainig_data_name=trainig_data_name, model_name=model_name, loss_name=loss_name)
    
    return model#, history

def model3_dloss(X_train, y_train, X_test, y_test, nx, ny, color, epochs, batch_size, loss_name, model_name, path, filter, trainig_data_name,  kl_divergence_function):

    # Making a model.
    """
    Parameters
    ----------
    KL_DIVERGENCE_FUNCTION : str
        Divergence function.
    X_TRAIN : ndarray
        Input image data before the CNN is trained.
    Y_TRAIN : ndarray
        Image labels for the CNN while it is being trained.
    X_TEST : ndarray
        Input image data after the CNN is trained it is tested with this dataset.
    Y_TEST : ndarray
        Image labels. After the CNN is trained it is tested with this dataset.
    EPOCH : int
        Number of training steps. The number of times to go through the full training data.
    BATCH_SIZE : int
        Number of training data required to update the model.
    LOSS_NAME : str
        The prob. loss function used duing the training process. It's probably a custom loss funtion.
        eg. "elbo" or "nll"
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    """
    model = Sequential()

    # Added Layers.
    model.add(tfp.layers.Convolution2DFlipout(filters = 64,
                                              input_shape=(nx, ny, color), kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , padding = 'valid',
                                              ))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(tfp.layers.Convolution2DFlipout(filters = 32, kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , padding = 'valid',
                                              ))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))
                          
    model.add(tfp.layers.Convolution2DFlipout(filters = 16, kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , padding = 'valid',
                                              ))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(tfp.layers.Convolution2DFlipout(filters = 8, kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , padding = 'valid',
                                              ))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(tfp.layers.Convolution2DFlipout(filters = 4, kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , padding = 'valid',
                                              ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(tfp.layers.DenseFlipout(1, activation="linear", kernel_divergence_fn=kl_divergence_function,))

    #for l in range(hp.Int('loss', 0, len(losses))):
    model.compile(optimizer= "adam", loss = loss_name, metrics=["mse"])

    #plotlosses = PlotLossesKeras()

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=epochs, batch_size=batch_size,
              #callbacks=[plotlosses],
              verbose=1)
    print("Model Training complete... ")

    # Visualize Model
    print(model.summary())

    loss = model.evaluate(X_test, y_test)
    
    # save model and model_weights
    print("Saving model and model weights ...")
    utils.save_model_modelweights(bcnn_model=model, path=path, filter=filter, trainig_data_name=trainig_data_name, model_name=model_name, loss_name=loss_name)
    
    return model#, history

def model4_dloss(X_train, y_train, X_test, y_test, nx, ny, color, epochs, batch_size, loss_name, model_name, path, filter, trainig_data_name,  kl_divergence_function):

    # Making a model.
    """
    Parameters
    ----------
    KL_DIVERGENCE_FUNCTION : str
        Divergence function.
    X_TRAIN : ndarray
        Input image data before the CNN is trained.
    Y_TRAIN : ndarray
        Image labels for the CNN while it is being trained.
    X_TEST : ndarray
        Input image data after the CNN is trained it is tested with this dataset.
    Y_TEST : ndarray
        Image labels. After the CNN is trained it is tested with this dataset.
    EPOCH : int
        Number of training steps. The number of times to go through the full training data.
    BATCH_SIZE : int
        Number of training data required to update the model.
    LOSS_NAME : str
        The prob. loss function used duing the training process. It's probably a custom loss funtion.
        eg. "elbo" or "nll"
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    """
    model = Sequential()

    # Added Layers.
    model.add(tfp.layers.Convolution2DFlipout(filters = 32,
                                              input_shape=(nx, ny, color), kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , padding = 'valid',
                                              ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(tfp.layers.Convolution2DFlipout(filters = 28, kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , padding = 'valid',
                                              ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dropout(0.2))
    model.add(tfp.layers.DenseFlipout(100, activation="linear", kernel_divergence_fn=kl_divergence_function,))
    
    model.add(Dropout(0.2))
    model.add(tfp.layers.DenseFlipout(10, activation="linear", kernel_divergence_fn=kl_divergence_function,))

    model.add(Dropout(0.2))
    model.add(tfp.layers.DenseFlipout(1, activation="linear", kernel_divergence_fn=kl_divergence_function,))

    #for l in range(hp.Int('loss', 0, len(losses))):
    model.compile(optimizer= "adam", loss = loss_name, metrics=["mse"])

    #plotlosses = PlotLossesKeras()

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=epochs, batch_size=batch_size,
              #callbacks=[plotlosses],
              verbose=1)
    print("Model Training complete... ")

    # Visualize Model
    print(model.summary())

    loss = model.evaluate(X_test, y_test)
    
    # save model and model_weights
    print("Saving model and model weights ...")
    utils.save_model_modelweights(bcnn_model=model, path=path, filter=filter, trainig_data_name=trainig_data_name, model_name=model_name, loss_name=loss_name)
    
    return model#, history



############################################################
# Models trained using the architecture in ML Paper I
############################################################

def old_model1(X_train, y_train, X_test, y_test, nx, ny, color, epochs, batch_size, loss_name, model_name, path, filter, trainig_data_name,  kl_divergence_function, wedge=False, training=False):

    # Making a model.
    """
    Parameters
    ----------
    KL_DIVERGENCE_FUNCTION : str
        Divergence function.
    X_TRAIN : ndarray
        Input image data before the CNN is trained.
    Y_TRAIN : ndarray
        Image labels for the CNN while it is being trained.
    X_TEST : ndarray
        Input image data after the CNN is trained it is tested with this dataset.
    Y_TEST : ndarray
        Image labels. After the CNN is trained it is tested with this dataset.
    EPOCH : int
        Number of training steps. The number of times to go through the full training data.
    BATCH_SIZE : int
        Number of training data required to update the model.
    LOSS_NAME : str
        The prob. loss function used duing the training process. It's probably a custom loss funtion.
        eg. "elbo" or "nll"
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    WEDGE : BOOL
        False if the data is not filtered and True if the data is filtered.
    TRAINING : BOOL
        if True the dropout will be active during to testing processes.
    """

    input_shape=(nx, ny, color)
    input0 = Input(shape=input_shape)
    inner = tfp.layers.Convolution2DFlipout(filters = 16,
                                              kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , #padding = 'valid',
                                              )(input0)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    
    inner = tfp.layers.Convolution2DFlipout(filters = 32,
                                              kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , #padding = 'valid',
                                              )(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    
    inner = tfp.layers.Convolution2DFlipout(filters = 64,
                                              kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , #padding = 'valid',
                                              )(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)
    
    if wedge == False:
        inner = tfp.layers.Convolution2DFlipout(filters = 256,
                                              kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , #padding = 'valid',
                                              )(inner)
        inner = BatchNormalization()(inner)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)

        inner = GlobalAveragePooling2D()(inner)
        
        inner = Dropout(0.2)(inner, training=training)
        inner = tfp.layers.DenseFlipout(350, activation='relu', kernel_divergence_fn=kl_divergence_function,)(inner)
    
    if wedge == True:
        inner = tfp.layers.Convolution2DFlipout(filters = 128,
                                              kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , #padding = 'valid',
                                              )(inner)
        inner = BatchNormalization()(inner)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)
        
        inner = tfp.layers.Convolution2DFlipout(filters = 128,
                                              kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , #padding = 'valid',
                                              )(inner)
        inner = BatchNormalization()(inner)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)

        inner = GlobalAveragePooling2D()(inner)

        inner = Dropout(0.2)(inner, training=training)
        inner = tfp.layers.DenseFlipout(250, activation='relu', kernel_divergence_fn=kl_divergence_function,)(inner)
    
    inner = Dropout(0.2)(inner, training=training)
    inner = tfp.layers.DenseFlipout(200, activation='relu', kernel_divergence_fn=kl_divergence_function,)(inner)
    
    inner = Dropout(0.2)(inner, training=training)
    inner = tfp.layers.DenseFlipout(100, activation='relu', kernel_divergence_fn=kl_divergence_function,)(inner)
    
    inner = Dropout(0.2)(inner, training=training)
    inner = tfp.layers.DenseFlipout(20, activation='relu', kernel_divergence_fn=kl_divergence_function,)(inner)
    
    output = tfp.layers.DenseFlipout(1, activation='linear', kernel_divergence_fn=kl_divergence_function,)(inner)
    
    model = Model(inputs=input0, outputs=output)
    
    # Compile Model
    model.compile(loss=loss_name,optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.))
    
    # Train model
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=epochs, batch_size=batch_size,
              verbose=1)
    print("Model Training complete... ")

    # Summary of model used
    print(model.summary())
    
    loss = model.evaluate(X_test, y_test)
    
    # save model and model_weights
    print("Saving model and model weights ...")
    utils.save_model_modelweights(bcnn_model=model, path=path, filter=filter, trainig_data_name=trainig_data_name, model_name=model_name, loss_name=loss_name)
    
    return model

def old_model2(X_train, y_train, X_test, y_test, nx, ny, color, epochs, batch_size, loss_name, model_name, path, filter, trainig_data_name,  kl_divergence_function, wedge=False, training=False):

    # Making a model.
    """
    Parameters
    ----------
    KL_DIVERGENCE_FUNCTION : str
        Divergence function.
    X_TRAIN : ndarray
        Input image data before the CNN is trained.
    Y_TRAIN : ndarray
        Image labels for the CNN while it is being trained.
    X_TEST : ndarray
        Input image data after the CNN is trained it is tested with this dataset.
    Y_TEST : ndarray
        Image labels. After the CNN is trained it is tested with this dataset.
    EPOCH : int
        Number of training steps. The number of times to go through the full training data.
    BATCH_SIZE : int
        Number of training data required to update the model.
    LOSS_NAME : str
        The prob. loss function used duing the training process. It's probably a custom loss funtion.
        eg. "elbo" or "nll"
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    WEDGE : BOOL
        False if the data is not filtered and True if the data is filtered.
    TRAINING : BOOL
        if True the dropout will be active during to testing processes.
    """
    input_shape=(nx, ny, color)
    input0 = Input(shape=input_shape)
    inner = tfp.layers.Convolution2DFlipout(filters = 16,
                                              kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , #padding = 'valid',
                                              )(input0)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    
    inner = tfp.layers.Convolution2DFlipout(filters = 32,
                                              kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , #padding = 'valid',
                                              )(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    
    inner = tfp.layers.Convolution2DFlipout(filters = 64,
                                              kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , #padding = 'valid',
                                              )(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)
    
    if wedge == False:
        inner = tfp.layers.Convolution2DFlipout(filters = 256,
                                              kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , #padding = 'valid',
                                              )(inner)
        inner = BatchNormalization()(inner)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)

        inner = GlobalAveragePooling2D()(inner)
        
        inner = Dropout(0.2)(inner, training=training)
        inner = tfp.layers.DenseFlipout(350, activation='relu', kernel_divergence_fn=kl_divergence_function,)(inner)
    
    if wedge == True:
        inner = tfp.layers.Convolution2DFlipout(filters = 128,
                                              kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , #padding = 'valid',
                                              )(inner)
        inner = BatchNormalization()(inner)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)
        
        inner = tfp.layers.Convolution2DFlipout(filters = 128,
                                              kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , #padding = 'valid',
                                              )(inner)
        inner = BatchNormalization()(inner)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)

        inner = GlobalAveragePooling2D()(inner)

        inner = Dropout(0.2)(inner, training=training)
        inner = tfp.layers.DenseFlipout(250, activation='relu', kernel_divergence_fn=kl_divergence_function,)(inner)
    
    inner = Dropout(0.2)(inner, training=training)
    inner = tfp.layers.DenseFlipout(200, activation='relu', kernel_divergence_fn=kl_divergence_function,)(inner)
    
    inner = Dropout(0.2)(inner, training=training)
    inner = tfp.layers.DenseFlipout(100, activation='relu', kernel_divergence_fn=kl_divergence_function,)(inner)
    
    inner = Dropout(0.2)(inner, training=training)
    inner = tfp.layers.DenseFlipout(20, activation='relu', kernel_divergence_fn=kl_divergence_function,)(inner)
    
    output = tfp.layers.DenseFlipout(1, activation='linear', kernel_divergence_fn=kl_divergence_function,)(inner)
    output = tfp.layers.DistributionLambda( lambda t: tfd.Normal(loc=t[..., :1],
                           #scale = 1)))
                           scale=1e-3 + tf.math.softplus(0.01 * t[..., :1])))(output)
    
    model = Model(inputs=input0, outputs=output)
    
    # Compile Model
    model.compile(loss=loss_name,optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.))
    
    # Train model
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=epochs, batch_size=batch_size,
              verbose=1)
    print("Model Training complete... ")

    # Summary of model used
    print(model.summary())
    
    loss = model.evaluate(X_test, y_test)
    
    # save model and model_weights
    print("Saving model and model weights ...")
    utils.save_model_modelweights(bcnn_model=model, path=path, filter=filter, trainig_data_name=trainig_data_name, model_name=model_name, loss_name=loss_name)
    
    return model



############################################################
# Hyperparameter Optimization Models
############################################################
def hp_model(hp):
    input_shape=(nx, ny, color)
    input0 = Input(shape=input_shape)
    inner = tfp.layers.Convolution2DFlipout(filters = 16,
                                              kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , #padding = 'valid',
                                              )(input0)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    
    inner = tfp.layers.Convolution2DFlipout(filters = 32,
                                              kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , #padding = 'valid',
                                              )(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    
    inner = tfp.layers.Convolution2DFlipout(filters = 64,
                                              kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , #padding = 'valid',
                                              )(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)
    
    # Number of hidden layers
    for i in range(hp.Int('num_layers', 0, 2)):
        #Note that we still test a different number of units for each layer.
        #There is a requirement that each Hyperparameter name should be unique.
        inner = tfp.layers.Convolution2DFlipout(filters = hp.Int('filters_' + str(i),
                                                                 min_value=16,
                                                                 max_value=64,
                                                                 step=16),
                                              kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , #padding = 'valid',
                                              )(inner)
        inner = BatchNormalization()(inner)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)
        
    inner = GlobalAveragePooling2D()(inner)
    
    inner = Dropout(0.2)(inner, training=training)
    inner = tfp.layers.DenseFlipout(units=hp.Int('units',min_value=100,max_value=200,step=50,default=100), activation='relu', kernel_divergence_fn=kl_divergence_function,)(inner)
    
    inner = Dropout(0.2)(inner, training=training)
    inner = tfp.layers.DenseFlipout(100, activation='relu', kernel_divergence_fn=kl_divergence_function,)(inner)
    
    inner = Dropout(0.2)(inner, training=training)
    inner = tfp.layers.DenseFlipout(20, activation='relu', kernel_divergence_fn=kl_divergence_function,)(inner)
    
    output = tfp.layers.DenseFlipout(1, activation='linear', kernel_divergence_fn=kl_divergence_function,)(inner)
    output = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1],
                           #scale = 1)))
                           scale=1e-3 + tf.math.softplus(0.01 * t[..., :1])))(output)
    
    model = Model(inputs=input0, outputs=output)
    
    # Compile Model
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.),
                  loss=loss_name,
                  metrics='mse')#=['r2_keras','mse', 'elbo','neg_log_likelihood', 'Mean_Squared_over_true_Error'])
    
    print(model.summary())
    return model



