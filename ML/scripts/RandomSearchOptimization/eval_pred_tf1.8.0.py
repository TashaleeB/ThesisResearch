#printf "\e[?2004l"
# needs to be ran in dp-GPU environment with Tensorflow version
# tf.__version__ : '1.8.0'

from __future__ import print_function, division, absolute_import

# Imports
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import json, os, sys, time, math, matplotlib, h5py, glob
import numpy as np
import tensorflow as tf
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

from matplotlib import cm
from matplotlib import gridspec
from sklearn.model_selection import KFold
from scipy.ndimage import gaussian_filter


import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import backend as K
#from keras.wrappers.scikit_learn import train_test_split

import gc
gc.enable()

tf.logging.set_verbosity(tf.logging.INFO)

data_path = '/pylon5/as5phnp/tbilling/data/'
reionfilename = data_path+'t21_snapshots_nowedge_v9.hdf5'
path_model = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/add_conv_layer/nowedge_v7/"
inputFile = reionfilename
outputdir = '/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/add_conv_layer/eval_pred_v9data/'

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

def makeModel(input_shape,Nregressparams):
    
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
    """
    Don't include this 2D pooling layer for better results.
    It makes sense that it does better without this layer because more of the feature map gets pushed through the network.
    """
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(GlobalAveragePooling2D())
    #I need to figureout
    #model.add(Dropout(0.2))
    #model.add(Dense(400, activation='relu'))
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

kfold_split = sorted(glob.glob("train_test_index_*.npz"))
models= sorted(glob.glob(path_model+"model_fold*.h5"))
scores =[]
istart = 0

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
                                     
                    
    print('loading model...')
    """
    print('making model...')
    print('compiling model...')
    print('number of regression parameters: ', trainlabels.shape[1])
    model = makeModel(input_shape, Nregressparams=trainlabels.shape[1])
    """
    model = load_model(models[i],custom_objects= {"Mean_Squared_over_true_Error":Mean_Squared_over_true_Error,"r2_keras":r2_keras})
    
    """
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
    print('saving model...')
    model.save(outputdir+"model_fold{}.h5".format(fold_num))
    model.save_weights(outputdir+"model_weights_fold{}.h5".format(fold_num))
    """
    # The evaluate() method - gets the loss statistics on already trained model using the validation (or test) data and the corresponding labels. Returns the loss value and metrics values for the model.
    print('calculating test loss...')
    score = model.evaluate(test_image, testlabels, batch_size=32, verbose=1) # 1 = progress bar
    # returns: loss
    print('          Test loss:', score)
    print('')
    #loss, mse, mae, mape
    scores.append(score)
    """
    # Save history
    print("Removing Scaling factor ({}) and saving histories...".format(factor))
    history_keys = np.array(list(history.history.keys()))
    for key in history_keys:
        np.savez(outputdir+"history_{}_{:d}".format(str(key),fold_num),
                 metric=np.array(history.history[str(key)])/factor)
    """
    #save predictions
    print('saving predictions for fold', fold+1, '...')
    if fold == 0:
        results = None
    savePreds(model, test_image, testlabels, len(trainlabels), fold=i, istart=istart, outdir=outputdir)
    istart += len(test_index)
        
    #destroy the current TF graph and creates a new one
    print("removing model history and TF graph...")
    #del history
    del model
    gc.collect()
    K.clear_session()

np.savez(outputdir+"eval_pred_score_{:d}_v9data".format(fold_num),scores=scores)
