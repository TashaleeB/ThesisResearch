# -*- coding: utf-8 -*-

import numpy as np
import h5py, glob

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

import gc
gc.enable()

inputFile = "/pylon5/as5phnp/tbilling/data/t21_snapshots_wedge_v9.hdf5"
#output = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/opt_versions/wedgefilter_v9/"
output = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/opt_versions/200steps_v9/"

#perfectmodel = "hyperParam_model_8.h5" # wedge
perfectmodel = "hyperParam_model_1.h5" # nowedge
factor = 1000.0
istart = 0

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
    # Create a custom loss function that divides the difference by the true
    #if not K.is_tensor(y_pred):
    #if not K.is_keras_tensor(y_pred):
    #    y_pred = K.constant(y_pred)

    y_true = K.cast(y_true, y_pred.dtype)  # Casts a tensor to a different dtype and returns it.
    diff_ratio = K.square((y_pred - y_true) / K.clip(K.abs(y_true), K.epsilon(), None))
    loss = K.mean(diff_ratio, axis=-1)
    # Return a function

    return loss


def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

#trainlabels = readLabels(ind=None)[:,5] * factor
#trainlabels = trainlabels.reshape(-1, 1)
#images, shape = readImages(ind=np.arange(1000))
kfold_split =sorted(glob.glob("/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/train_test_index_*.npz"))

for fold in range(len(kfold_split)):

    model = load_model(perfectmodel,
    custom_objects={"Mean_Squared_over_true_Error": Mean_Squared_over_true_Error,
    "r2_keras": r2_keras})

    print(model.summary())

    # try to evaluate the model for a few images
    # Load Index Label
    test_index = np.load(kfold_split[fold])["test_index"]
    
    # Load Test Labels and Test Images
    test_labels = readLabels(ind=None)[test_index,5]*factor
    test_labels = test_labels.reshape(-1, 1)
    test_images,input_shape = readImages(ind=test_index)
    
    #test_images = images[:100, :, :, :]
    #test_labels = trainlabels[:100, :]

    #save predictions
    print('saving predictions for fold', fold+1, '...')
    eval_data, eval_labels, Ntot = test_images, test_labels, len(test_labels)

    # The Predict() method -  is for the actual prediction. It generates output predictions for the input samples.
    preds = model.predict(eval_data, verbose=0).flatten() #0 = silent

    Nregressparams = len(eval_labels[0])

    results = np.zeros((Ntot, Nregressparams),
                           dtype = [('truth', 'f'), ('prediction', 'f'), ('fold', 'i')])
    iend = istart+len(eval_labels)

    #print('istart and iend', istart, iend)

    results['fold'] = fold
    results['truth'] = eval_labels
    #results['truth'][istart-100:iend-100] = eval_labels
    for n in range(Nregressparams):
        results['prediction'][:,n] = preds[n::Nregressparams]
        #results['prediction'][istart-100:iend-100,n] = preds[n::Nregressparams]

    np.save("bestmodel_pred_results_{}".format(fold), results)
    #istart += len(test_labels)
    #print("Rotating images and labels")
    #images = np.roll(images,-100)
    #trainlabels = np.roll(trainlabels,-100)
    
    #destroy the current TF graph and creates a new one
    print("removing model history and TF graph...")
    del model
    gc.collect()
    K.clear_session()
