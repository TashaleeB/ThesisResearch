#printf "\e[?2004l"
# needs to be ran in hp_opt environment with Tensorflow version
# tf.__version__ : '2.0.0'
# anything related to optimization you should use this version of tf.

from __future__ import print_function, division, absolute_import

# Imports
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import h5py, glob
import numpy as np, matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K


import gc
gc.enable()


data_path = '/pylon5/as5phnp/tbilling/data/'
reionfilename = data_path+'t21_snapshots_nowedge_v12.hdf5'
path_model = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/opt_versions/200steps/"
inputFile = reionfilename
outputdir = path_model

steps = 400
factor =1000.
nfold = 10
fold_num = 0

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

def savePreds(model, eval_data, eval_labels, Ntot, fold, istart=0, outdir=None):
    outputFile = os.path.join(outdir, "eval_pred_results{:d}_v9data.npy".format(fold))

    # The Predict() method -  is for the actual prediction. It generates output predictions for the input samples.
    preds = model.predict(eval_data, verbose=0).flatten() #0 = silent

    Nregressparams = len(eval_labels[0])

    results = np.zeros((Ntot, Nregressparams),
                           dtype = [('truth', 'f'), ('prediction', 'f'), ('fold', 'i')])
    iend = istart+len(eval_labels)

    #print('istart and iend', istart, iend)

    results['fold'][istart:iend] = fold
    results['truth'][istart:iend] = eval_labels
    #results['truth'][istart-100:iend-100] = eval_labels
    for n in range(Nregressparams):
        results['prediction'][istart:iend,n] = preds[n::Nregressparams]
        #results['prediction'][istart-100:iend-100,n] = preds[n::Nregressparams]

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

kfold_split = sorted(glob.glob("/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/train_test_index_*.npz"))
models= sorted(glob.glob(path_model+"hyperParam_model_*.h5"))[:-1]
scores =[]
istart = 0

for i in np.arange(len(kfold_split)):
    # Load Index Label
    train_index = np.load(kfold_split[i])["train_index"]
    test_index = np.load(kfold_split[i])["test_index"]
    # Load Labels and Images
    trainlabels = readLabels(ind=None)[train_index,5]*factor
    trainlabels = trainlabels.reshape(-1, 1)
    images,shape_label = readImages(ind=train_index)
    
    testlabels = readLabels(ind=None)[test_index,5]*factor
    testlabels = testlabels.reshape(-1, 1)
    testimages,input_shape = readImages(ind=test_index)

    fold = i
    log_dir = os.path.join(outputdir, 'output', str(fold))
    cb = keras.callbacks.TensorBoard(log_dir=log_dir,
                                     histogram_freq=10, write_images=True)
                                     
                    
    print('loading and compiling model...')
    
    model = load_model(models[i],custom_objects= {"Mean_Squared_over_true_Error":Mean_Squared_over_true_Error,"r2_keras":r2_keras}, compile=False)
    # Only complie if it wasn't included in the saved model
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.),
    loss=Mean_Squared_over_true_Error,
    metrics=[r2_keras,'mse', 'mae', 'mape'])
    
    #save predictions
    print('saving predictions for fold', fold+1, '...')
    savePreds(model, testimages, testlabels, len(trainlabels), fold=i, istart=istart, outdir=outputdir)
    istart += len(test_index)
        
    #destroy the current TF graph and creates a new one
    print("removing model history and TF graph...")
    #del history
    del model
    gc.collect()
    K.clear_session()

trainlabels = readLabels(ind=None)[:,5] * factor
trainlabels = trainlabels.reshape(-1, 1)
images, shape = readImages(ind=np.arange(1000))

scores=[]
param_count = []
models = sorted(glob.glob("200steps/hyperParam_model_*.h5"))[:-1]
