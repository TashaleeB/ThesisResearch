# -*- coding: utf-8 -*-

import numpy as np, matplotlib.pyplot as plt
import h5py, glob

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

import gc
gc.enable()

inputFile = "/pylon5/as5phnp/tbilling/data/t21_snapshots_nowedge_v9.hdf5"
#output = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/opt_versions/wedgefilter_v9/"
output = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/opt_versions/200steps_v9/"

#perfectmodel = "hyperParam_model_8.h5" # wedge
perfectmodel = "hyperParam_model_1.h5" # nowedge

factor = 1000.0
istart = 0

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

    if 'crop' in params:
        #use just the top corner of the images
        data = np.asarray(f['Data'][u't21_snapshots'][ind, :, 0 : params['crop'], 0 : params['crop']])
    else:
        #use everything!
        print('reading all data', len(ind))
        data = np.asarray(f['Data'][u't21_snapshots'][ind, :, :, :]) # (N_realizations, N_redshifts, N_pix, N_pix)
        #print('loaded data', len(ind))

    print('finished loading data.', data.shape)

    data  = data.transpose(0,2,3,1)  # (N_realizations, N_pix, N_pix, N_redshifts)

    return data, data[0].shape

def Mean_Squared_over_true_Error(y_true, y_pred):

    y_true = K.cast(y_true, y_pred.dtype)  # Casts a tensor to a different dtype and returns it.
    diff_ratio = K.square((y_pred - y_true) / K.clip(K.abs(y_true), K.epsilon(), None))
    loss = K.mean(diff_ratio, axis=-1)
    # Return a function

    return loss

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


# load the model
pmodel = output+perfectmodel
model = load_model(pmodel,
custom_objects={"Mean_Squared_over_true_Error": Mean_Squared_over_true_Error,
"r2_keras": r2_keras})

# define model_ to output right after the first hidden layer
model_ = Model(inputs=model.inputs, outputs=model.layers[1].output)
model_.summary()

# load the images
trainlabels = readLabels(ind=None)[:,5] * factor
trainlabels = trainlabels.reshape(-1, 1)
images, shape = readImages(ind=np.arange(1000))

# get feature map for first hidden layer
feature_maps = model_.predict(images)

# plot all 64 maps in an 8x8 squares
square = 10
ix = 1

plt.figure(figsize=(15,10))
for indx in [0, 3, 6, 9, 12]: #[0, 3, 6, 9]:
    model_ = Model(inputs=model.inputs, outputs=model.layers[indx].output)
    feature_maps = model_.predict(images)
    
    for _ in range(square):
        # specify subplot and turn of axis
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        plt.imshow(feature_maps[0, :, :, ix-1])
        ix += 1
# show the figure
plt.subplots_adjust(bottom=0.1, wspace=0.0, hspace=0.0)
#plt.subplots_adjust(bottom=0.0, wspace=-0.8, hspace=0.0)
#plt.tight_layout()
plt.show()
