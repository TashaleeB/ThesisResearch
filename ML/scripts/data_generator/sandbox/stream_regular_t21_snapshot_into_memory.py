# needs to be ran in dl-gpu environment with Tensorflow version
# tf.__version__ : '1.8.0'

from __future__ import print_function, division, absolute_import

# Imports
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import os, sys, matplotlib, h5py, glob
import numpy as np
import keras
import tensorflow as tf
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

from matplotlib import cm
from matplotlib import gridspec
from sklearn.model_selection import KFold
from scipy.ndimage import gaussian_filter


from keras import backend as K
from keras import preprocessing
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
#from keras.wrappers.scikit_learn import train_test_split

import gc
gc.enable()

data_path = '/pylon5/as5phnp/tbilling/data/'
reionfilename = data_path+'t21_snapshots_nowedge_v7.hdf5'
outputdir = '/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/'
inputFile = reionfilename

factor =1000.
batch_size = 32
num_classes = 1
epochs = 150
n = 100
istart = 0

kfold_split = outputdir+"train_test_index_0.npz"
scores =[]
istart = 0

# Load Index Label
train_index = np.load(kfold_split)["train_index"]
test_index = np.load(kfold_split)["test_index"]

x_train = np.asarray(h5py.File(inputFile, 'r')['Data'][u't21_snapshots'][train_index,:,:,:]).transpose(0,2,3,1) #(N_realizations, N_pix, N_pix, N_redshifts)
x_test = np.asarray(h5py.File(inputFile, 'r')['Data'][u't21_snapshots'][test_index,:,:,:]).transpose(0,2,3,1) #(N_realizations, N_pix, N_pix, N_redshifts)

y_train = (np.asarray(h5py.File(inputFile, 'r')['Data'][u'snapshot_labels'][train_index, 5])*factor).reshape(-1, 1)
y_test = (np.asarray(h5py.File(inputFile, 'r')['Data'][u'snapshot_labels'][test_index, 5])*factor).reshape(-1, 1)

def Mean_Squared_over_true_Error(y_true, y_pred):
    # Create a custom loss function that divides the difference by the true
    #if not K.is_tensor(y_pred):
    if not K.is_keras_tensor(y_pred):
        y_pred = K.constant(y_pred)

    y_true = K.cast(y_true, y_pred.dtype)
    diff_ratio = K.square((y_pred - y_true)/K.clip(K.abs(y_true),K.epsilon(),None))
    loss = K.mean(diff_ratio, axis=-1)
    # Return a function

    return loss

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def makeModel(input_shape):

    model = Sequential()

    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu',input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
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

    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.),
               loss=Mean_Squared_over_true_Error)
    #plot_model(model, to_file="images/model_plot.png", show_shapes=True, show_layer_names=True)
    print(model.summary())
    return model

datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format="channels_last",
    validation_split=0.0,
    dtype=None,
)
    
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

# Build model
print('making model...')
print('compiling model...')
print('number of regression parameters: ', num_classes)
model = makeModel(input_shape=x_train.shape[1:])

# here's a more "manual" example
for e in range(epochs):
    print("*"*50)
    print('Epoch', e)
    print("*"*50)
    batches = 1
    for i in range(0, len(x_train), n): #Feed in 21cm data in groups of n at a time
        y_train_ = y_train[i:i + n]
        x_train_ = x_train[i:i + n]
        for x_batch, y_batch in datagen.flow(x_train_, y_train_, batch_size=batch_size): # break the n samples in to a batch
            history = model.fit(x_batch, y_batch, validation_split=0.2) # Train and Validate batches of data aka incremental learning.
            print("Batch number",batches)
            batches += 1
            if batches >= ((len(x_train_)/batch_size) -1):
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

# Save Model
model.save(outputdir+"model_stream_21cmSnapshots_in_memory.h5")
print("Saving ... "+outputdir+"model_stream_21cmSnapshots_in_memory.h5")

# Save history
print("Removing Scaling factor and saving history...")
history_keys = np.array(list(history.history.keys()))
for key in history_keys:
   np.savez(outputdir+"history_stream_21cmSnapshots_in_memory",
            metric=np.array(history.history[str(key)])/factor)


# Evaluate the already trained model
score=model.evaluate(x_test, y_test, batch_size=32)
scores.append(score)

#save predictions
print('saving predictions for fold', fold+1, '...')
eval_data, eval_labels, Ntot = x_test, y_test, len(y_test)

# The Predict() method -  is for the actual prediction. It generates output predictions for the input samples.
preds = model.predict(eval_data, verbose=0).flatten() #0 = silent

Nregressparams = len(eval_labels[0])

results = np.zeros((Ntot, Nregressparams),
                       dtype = [('truth', 'f'), ('prediction', 'f'), ('fold', 'i')])
iend = istart+len(eval_labels)

#print('istart and iend', istart, iend)

results['fold'] = fold
results['truth'] = eval_labels

for i in range(Nregressparams):
    results['prediction'][:,i] = preds[i::Nregressparams]

np.save(outputdir+"eval_pred_results_stream_21cmSnapshots_in_memory", results)
print("Results: ", results)
istart += len(test_labels)

print("Save scores...")
np.savez(outputdir+"eval_pred_score_stream_21cmSnapshots_in_memory",scores=scores)
