from __future__ import print_function, division, absolute_import

# Imports
import json, os, sys, time, math, matplotlib, h5py, glob
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
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hypermodel import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters

data_path = '/pylon5/as5phnp/tbilling/data/'
reionfilename = data_path+'t21_snapshots_nowedge_v9.hdf5'
inputFile = reionfilename
outputdir = '/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/'

steps = 400
factor =1000.
nfold = 10

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

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def savePreds(model, eval_data, eval_labels, Ntot, fold, istart=0, outdir=None):
    outputFile = os.path.join(outdir, "results{:d}.npy".format(fold))

    # The Predict() method -  is for the actual prediction. It generates output predictions for the input samples.
    preds = model.predict(eval_data, verbose=0).flatten() #0 = silent

    Nregressparams = len(eval_labels[0])

    results = np.zeros((Ntot, Nregressparams),
                           dtype = [('truth', 'f'), ('prediction', 'f'), ('fold', 'i')])
    iend = istart+len(eval_labels)

    #print('istart and iend', istart, iend)

    results['fold'][istart:iend] = fold
    results['truth'][istart:iend] = eval_labels
    for n in range(Nregressparams):
        results['prediction'][istart:iend,n] = preds[n::Nregressparams]

    np.save(outputFile, results)

params = {'runlabel':'alldata_b_',
        'Nfolds': nfold,
        'debug': False,
        'epochs': steps,
        'crop': 512,
        'predicttwoparams': [1],# [0, 1],
        'patience': 20,
        'learning_rate': 0.1,
        'decay': True}

kfold_split = sorted(glob.glob("train_test_index_*.npz"))
scores =[]
istart = 0

for i in np.arange(nfold):
    # Load Index Label
    train_index = np.load(kfold_split[i])["train_index"]
    test_index = np.load(kfold_split[i])["test_index"]
    # Load Labels and Images
    trainlabels = readLabels(ind=None)[train_index,5]*factor
    trainlabels = trainlabels.reshape(-1, 1)
    images,shape = readImages(ind=train_index)
    
    testlabels = readLabels(ind=None)[test_index,5]*factor
    testlabels = testlabels.reshape(-1, 1)
    test_image,shape = readImages(ind=test_index)

    fold = i
    log_dir = os.path.join(outputdir, 'output', str(fold))
    cb = keras.callbacks.TensorBoard(log_dir=log_dir,
                                     histogram_freq=10, write_images=True)

    print('number of regression parameters: ', trainlabels.shape[1])
    #model = load_model(outputdir+'hyperParam_model.h5')
    model = load_model(outputdir+'hyperParam_model.h5',custom_objects={"r2_keras":r2_keras})

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
    # save model
    model.save(outputdir+"model_fold{}.h5".format(fold))
    model.save_weights(outputdir+"model_weights_fold{}.h5".format(fold))

    # The evaluate() method - gets the loss statistics on already trained model using the validation (or test) data and the corresponding labels. Returns the loss value and metrics values for the model.
    print('calculating test loss...')
    score = model.evaluate(test_image, testlabels, batch_size=32, verbose=1) # 1 = progress bar
    # returns: loss
    print('          Test loss:', score)
    print('')
    #loss, mse, mae, mape
    scores.append(score)
    
    # Save history
    print("Removing Scaling factor ({}) and saving histories...".format(factor))
    history_keys = np.array(list(history.history.keys()))
    for key in history_keys:
        np.savez(outputdir+"history_{}_{:d}".format(str(key),fold),
                 metric=np.array(history.history[str(key)])/factor)

    #save predictions
    print('saving for fold', fold+1, '...')
    if fold == 0:
        results = None
    savePreds(model, test_image, testlabels, len(trainlabels), fold, istart=istart, outdir=outputdir)
    istart += len(test_index)
    print("results saved...")
        
    #destroy the current TF graph and creates a new one
    print("removing model history and TF graph...")
    del history
    del model
    K.clear_session()

np.savez(outputdir+"score_{:d}".format(fold),scores=scores)
