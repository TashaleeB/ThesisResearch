from __future__ import print_function, division, absolute_import

# Imports
import json, os, sys, time, math, matplotlib, h5py, random, glob
import numpy as np

#random seed to control the reproducability
seed = 8675309
np.random.seed(seed)
#np.random.seed(datetime.datetime.now().microsecond)

import tensorflow as tf
#import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

from matplotlib import cm
from matplotlib import gridspec
from sklearn.model_selection import KFold
from scipy.ndimage import gaussian_filter

from tensorflow import keras

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import backend as K

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hypermodel import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters

data_path = '/pylon5/as5phnp/tbilling/data/'
reionfilename = data_path+'t21_snapshots_nowedge_v9.hdf5'
project_name='hp_opt'
outputdir = '/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/'
inputFile = reionfilename

steps = 5
factor =1000.
nfold = 10

min = 0
max = 999

#Select 400 random numbers from 0-999
digits = np.array([(random.randint(min, max)) for i in range(400)])
#np.savez(outputdir+"hyperParam_digits", digits=digits)


hp = HyperParameters()

def readLabels(ind=None, **params):
    """
    read in labels only
    (to use with batches """
    f = h5py.File(inputFile, 'r')

    if ind is None:
        labels = np.asarray(f['Data'][u'snapshot_labels'])  #(N_realizations, N_parameters)
    else:
        labels = np.asarray(f['Data'][u'snapshot_labels'][:, ind])

    if labels.ndim == 1:
        print('training on just one param.')
        print('starting with the following shape, dim:', labels.shape, labels.ndim)
        if labels.ndim > 1:
            labels = labels[:, params['predictoneparam']]
    elif labels.shape[1] == 2:
        print('training on two params.')
        print('starting with the following shape, dim:', labels.shape, labels.ndim)
        if labels.ndim > 1:
            labels = labels[:, ind]

    #if there's only one label per image, we'll have to reshape it:
    if labels.ndim == 1:
        print('reshaping data...')
        labels = labels.reshape(-1, 1)

    return labels

def readImages(ind, **params):
    """ 
    read in data
    """

    print('reading data from', inputFile)

    f = h5py.File(inputFile, 'r')

    #if params['debug'] == True:
    #    data = np.asarray(f['Data'][u't21_snapshots'][ind,:,0:16,0:16])
    if 'crop' in params:
        #print('cropping.')
        #use just the top corner of the images
        data = np.asarray(f['Data'][u't21_snapshots'][ind,:,0:params['crop'],0:params['crop']])
    else:
        #use everything!
        print('reading all data', len(ind))
        data = np.asarray(f['Data'][u't21_snapshots'][ind,:,:,:]) # (N_realizations, N_redshifts, N_pix, N_pix)
        #print('loaded data', len(ind))

    print('finished loading data.', data.shape)

    data  = data.transpose(0,2,3,1) #(N_realizations, N_pix, N_pix, N_redshifts)

    return data, data[0].shape

def Mean_Squared_over_true_Error(y_true, y_pred):
    # Create a custom loss function that divides the difference by the true
    #if not K.is_tensor(y_pred):
    #if not K.is_keras_tensor(y_pred):
    #    y_pred = K.constant(y_pred)

    y_true = K.cast(y_true, y_pred.dtype) #Casts a tensor to a different dtype and returns it.
    diff_ratio = K.square((y_pred - y_true)/K.clip(K.abs(y_true),K.epsilon(),None))
    loss = K.mean(diff_ratio, axis=-1)
    # Return a function

    return loss


def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

trainlabels = readLabels(ind=None)[digits,5]*factor
trainlabels = trainlabels.reshape(-1, 1)
images,shape =readImages(ind=np.arange(1000))
images =images[sorted(digits)]

def build_model(hp):
    
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu',input_shape=images.shape[1:]))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4,1e-5])),
                  loss=Mean_Squared_over_true_Error,
                  metrics=[r2_keras,'mse', 'mae', 'mape','msle'])
    #plot_model(model, to_file="images/model_plot.png", show_shapes=True, show_layer_names=True)
    print(model.summary())
    return model

# Start Tuning based on low validation loss
tuner = RandomSearch(build_model,
                     objective='val_loss', # 'loss', 'val_loss', 'val_accuracy'
                     max_trials=20,
                     executions_per_trial=1,
                     directory=outputdir,
                     project_name=project_name)

# Print a summary of the search space
tuner.search_space_summary()

# Show the best models, their hyperparameters, and the resulting metrics.
x = images[:360]
y = trainlabels[:360]
val_x = images[360:]
val_y = trainlabels[360:]

tuner.search(x, y, epochs=steps, validation_data=(val_x, val_y))
tuner.results_summary()

# Retrieve the best model.
best_model = tuner.get_best_models(num_models=1)[0]

# Save best hyperperam Model
print("saving best hyperparameter model ...")
best_model.save(outputdir+"hyperParam_model.h5")

# Save best hyperperam Model Weights
print("saving best hyperparameter model weights ...")
best_model.save_weights(outputdir+"hyperParam_model_weights.h5")



kfold_split = sorted(glob.glob("train_test_index_*.npz"))
scores =[]

for i in np.arange(nfold):
    # Load Index Label
    train_index = np.load(kfold_split[i])["train_index"]
    test_index = np.load(kfold_split[i])["test_index"]
    # Load Labels and Images
    trainlabels = readLabels(ind=None)[train_index,5]*factor
    trainlabels = trainlabels.reshape(-1, 1)
    images,shape_label = readImages(ind=train_index)

    testlabels = readLabels(ind=None)[test_index,5]*factor
    testlabels = testlabels.reshape(-1, 1)
    test_image,input_shape = readImages(ind=test_index)

    fold = i
    log_dir = os.path.join(outputdir, 'output', str(fold))
    cb = keras.callbacks.TensorBoard(log_dir=log_dir,
                                     histogram_freq=10, write_images=True)


    print('making model...')
    print('compiling model...')
    print('number of regression parameters: ', trainlabels.shape[1])
    model = makeModel(input_shape, Nregressparams=trainlabels.shape[1])

    #model = load_model(outputdir+'hyperParam_model.h5')
    #model = load_model(outputdir+'hyperParam_model.h5',custom_objects={"r2_keras":r2_keras})

    # The fit() method - Trains the model with the given inputs (and corresponding training labels)
    print('fitting model...')
    print("-"*150)
    print("*"*150)
    print("-"*150)
    #print start time
    os.system("date")
    history = model.fit(images, trainlabels, batch_size=32, verbose=2,
                        validation_split=0.2, epochs=steps, callbacks=[cb])
    os.system("date")
    print("-"*150)
    print("*"*150)
    print("-"*150)

