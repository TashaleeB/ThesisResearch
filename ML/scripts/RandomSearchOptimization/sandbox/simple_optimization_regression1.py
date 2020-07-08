from __future__ import print_function, division, absolute_import

# Imports
import json, os, sys, time, math, matplotlib, h5py
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
from sklearn.model_selection import GridSearchCV
from scipy.ndimage import gaussian_filter

from tensorflow import keras

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hypermodel import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters

data_path = '/pylon5/as5phnp/tbilling/data/'
reionfilename = data_path+'t21_snapshots_nowedge_v7.hdf5'
inputdir = '/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/wedgecut/'
inputFile = reionfilename
outputdir = inputdir

hp = HyperParameters()

def readLabels(ind=None, **params):
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

labels = readLabels(ind=None)[:,5]
labels = labels.reshape(-1, 1)
images,shape =readImages(ind=np.arange(1000))

def build_model(hp):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu',
                     input_shape=images.shape[1:]))
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

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4])),
              loss='mse')
              
    print(model.summary())
    return model

tuner = RandomSearch(build_model,
                     objective='val_loss', # 'loss', 'val_loss', 'val_accuracy'
                     max_trials=5,
                     executions_per_trial=3,
                     directory='/pylon5/as5phnp/tbilling/sandbox/',
                     project_name='PleaseWork')

"""
    # smaller dataset
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
"""

# Display search overview.
tuner.search_space_summary()

# Performs the hypertuning.
tuner.search(images, labels, epochs=2, validation_split=0.1)
"""
New version
    val_x=images[digits][0:40]
    val_y=trainlabels[0:40]
    tuner.search(images[digits][41:], trainlabels[41:], epochs=30, validation_data=(val_x, val_y))
"""

# Show the best models, their hyperparameters, and the resulting metrics.
tuner.results_summary()

# Retrieve the best model.
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model.
#loss, accuracy = best_model.evaluate(test_images, test_label)
#print('loss:', loss)
#print('accuracy:', accuracy)

# Save best hyperperam Model
print("saving best hyperparameter model ...")
best_model.save(outputdir+"model.h5")

# Save best hyperperam Model Weights
print("saving best hyperparameter model weights ...")
best_model.save_weights(outputdir+"model_weights.h5")

# returns a compiled model
# identical to the previous one
model = load_model(outputdir+"model.h5")
model_weights = some_model_atttribut_name.load_weights(outputdir+"model_weights.h5")

# Store History Variables
history = best_model.fit(images[TrainIndex_7], trainlabels, batch_size=32, verbose=2,
                         validation_split=0.1, epochs=2)
# Save history
np.savez(outputdir+"history".format(steps,fold),
         val_loss=np.array(history.history['val_loss']),
         loss=np.array(history.history['loss']))
