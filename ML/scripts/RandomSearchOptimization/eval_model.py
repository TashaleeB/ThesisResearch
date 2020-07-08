# -*- coding: utf-8 -*-

import numpy as np
import h5py, glob

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

import gc
gc.enable()

inputFile = "/pylon5/as5phnp/tbilling/data/t21_snapshots_nowedge_v7.hdf5"
output = "wedgefilter_v7/"

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
"""
trainlabels = readLabels(ind=None)[:,5] * factor
trainlabels = trainlabels.reshape(-1, 1)
images, shape = readImages(ind=np.arange(1000))

target_model = "200steps/hyperParam_model_10.h5"

model = load_model(
    target_model,
    custom_objects={
        "Mean_Squared_over_true_Error": Mean_Squared_over_true_Error,
        "r2_keras": r2_keras,
    }
)

# try to evaluate the model for a few images
test_images = images[:10, :, :, :]
test_labels = trainlabels[:10, :]

model.evaluate(test_images, test_labels, batch_size=32)
"""

trainlabels = readLabels(ind=None)[:,5] * factor
trainlabels = trainlabels.reshape(-1, 1)
images, shape = readImages(ind=np.arange(1000))

scores=[]
param_count = []
models = sorted(glob.glob(output+"hyperParam_model_*.h5"))[:-1]

for fold, m in enumerate(models):
    model = load_model(
    m,
    custom_objects={
        "Mean_Squared_over_true_Error": Mean_Squared_over_true_Error,
        "r2_keras": r2_keras,
        }
    )
    # You will need to compile for certain models
    #model.compile(optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.),
    #loss=Mean_Squared_over_true_Error,
    #metrics=[r2_keras,'mse', 'mae', 'mape'])

    print(model.summary())

    # try to evaluate the model for a few images
    test_images = images[:100, :, :, :]
    test_labels = trainlabels[:100, :]

    # Evaluate the already trained model
    score=model.evaluate(test_images, test_labels, batch_size=32)
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

    np.save(output+"eval_pred_results_{}".format(fold), results)
    print("Results: ", results)
    istart += len(test_labels)

    print("Save scores...")

    np.savez(output+"eval_pred_score_{:d}data".format(fold),scores=scores)
    scores.append(score[0])
    param_count.append(model.count_params())

    #destroy the current TF graph and creates a new one
    print("removing model history and TF graph...")
    #del history
    del model
    gc.collect()
    K.clear_session()

# Make Plots
plt.figure(figsize=(15,10))
plt.plot(np.array(param_count)[np.array(param_count).argsort()]/10000., np.array(scores)[np.array(param_count).argsort()],"-o")
plt.ylabel("Loss")
plt.xlabel("Model Complexity")
plt.xlim(0,100)
plt.savefig("200steps/model_complexity_1.png")
plt.show()

plt.figure(figsize=(10,15))
plt.plot(np.array(scores)[np.array(param_count).argsort()],"-o")
plt.ylabel("Loss")
plt.xlabel("Model Complexity")
plt.savefig("200steps/model_complexity_2.png")
plt.show()

np.array(scores)[np.array(param_count).argsort()]
     
np.savez("200steps/eval_pred_score_v7data",scores=scores)
np.savez("200steps/eval_pred_param_count_v7data",param_count=param_count)
np.savez(outputdir+"eval_pred_score_{:d}_v9data".format(fold_num),scores=scores)
