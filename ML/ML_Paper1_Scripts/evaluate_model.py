# -*- coding: utf-8 -*-

import numpy as np, matplotlib.pyplot as plt
import h5py, glob

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

import gc
gc.enable()

inputFile = "/pylon5/as5phnp/tbilling/data/t21_snapshots_wedge_v9.hdf5"
output = "wedgefilter_v9/"

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

trainlabels = readLabels(ind=None)[:,5] * factor
trainlabels = trainlabels.reshape(-1, 1)
images, shape = readImages(ind=np.arange(1000))

sorted_model_list = []

models = sorted(glob.glob(output+"hyperParam_model_*.h5"))[:-1]

for s in range(len(models)):
    sorted_model_list.append(output+'hyperParam_model_{}.h5'.format(s+1))

sorted_model_list= np.array(sorted_model_list)

plt.figure(figsize=(15,10))

for fold in range(len(sorted_model_list)):

    scores=[]
    param_count = []
    
    for m in sorted_model_list:
        model = load_model(
        m,
        custom_objects={
            "Mean_Squared_over_true_Error": Mean_Squared_over_true_Error,
            "r2_keras": r2_keras,
            }
        )

        print(model.summary())

        # try to evaluate the model for a few images
        test_images = images[:100, :, :, :]
        test_labels = trainlabels[:100, :]

        # Evaluate the olready trained model
        score=model.evaluate(test_images, test_labels, batch_size=32)
        
        scores.append(score[0])
        param_count.append(model.count_params())

        #destroy the current TF graph and creates a new one
        print("removing model history and TF graph...")
        #del history
        del model
        gc.collect()
        K.clear_session()
    
    print("Rotating images and labels")
    images = np.roll(images,-len(trainlabels)//len(sorted_model_list))
    trainlabels = np.roll(trainlabels,-len(trainlabels)//len(sorted_model_list))
    
    print("Adding to Plot...")
    plt.plot(np.array(param_count)[np.array(param_count).argsort()]/10000.,
            np.array(scores)[np.array(param_count).argsort()],"-o", label="Fold "+str(fold+1))
    print("Saving evaluation scores and parameter count...")
    np.savez(output+"evaluation_modelcomplex_{:d}data".format(fold+1),scores=scores, param_count=param_count)

plt.ylabel("Loss")
plt.xlabel("Model Complexity")
plt.xlim(0,100)
plt.legend()
plt.savefig(output+"model_complexity.png")
plt.show()

print("Ordered Number of Parameters:", np.array(param_count)[np.array(param_count).argsort()])
print("\n")
print("Ordered Scores:", np.array(scores)[np.array(param_count).argsort()])

print(sorted_model_list[np.array(param_count).argsort()])

"""
# Make Plots
plt.figure(figsize=(15,10))
plt.plot(np.array(param_count)[np.array(param_count).argsort()]/10000., np.array(scores)[np.array(param_count).argsort()],"-o")
plt.ylabel("Loss")
plt.xlabel("Model Complexity")
plt.xlim(0,100)
plt.savefig(output+"model_complexity_1.png")
plt.show()

plt.figure(figsize=(10,15))
plt.plot(np.array(scores)[np.array(param_count).argsort()],"-o")
plt.ylabel("Loss")
plt.xlabel("Model Complexity")
plt.savefig(output+"model_complexity_2.png")
plt.show()
"""
