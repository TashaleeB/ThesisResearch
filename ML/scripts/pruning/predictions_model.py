# -*- coding: utf-8 -*-

import numpy as np
import h5py, glob

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

import gc
gc.enable()

inputFile = "/lustre/aoc/projects/hera/tbilling/ml/data/t21_snapshots_wedge_v9.hdf5"
outputdir = "/lustre/aoc/projects/hera/tbilling/ml/pruning/"

train_test_file = "/lustre/aoc/projects/hera/tbilling/ml/data/train_test_index_80_20_split.npz"

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

# Load Index Label
test_index = np.load(train_test_file)["test_index"]

# Load images and labels for training and testing
test_labels = readLabels(ind=None)[test_index,5]*factor
test_labels = test_labels.reshape(-1, 1)
test_images,input_shape = readImages(ind=test_index)

def model():
    input0 = Input(shape=input_shape)
    #inner = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu',input_shape=images.shape[1:])(input0)
    inner = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu')(input0)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    
    inner = Conv2D(32, kernel_size=(3, 3), activation='relu')(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    
    inner = Conv2D(64, kernel_size=(3, 3), activation='relu')(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)
    
    if wedge == False:
        inner = Conv2D(256, kernel_size=(3, 3), activation='relu')(inner)
        inner = BatchNormalization()(inner)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)

        inner = GlobalAveragePooling2D()(inner)
        
        inner = Dropout(0.2)(inner, training=training)
        inner = Dense(350, activation='relu')(inner)
    
    else:
        inner = Conv2D(128, kernel_size=(3, 3), activation='relu')(inner)
        inner = BatchNormalization()(inner)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)
        
        inner = Conv2D(128, kernel_size=(3, 3), activation='relu')(inner)
        inner = BatchNormalization()(inner)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)

        inner = GlobalAveragePooling2D()(inner)

        inner = Dropout(0.2)(inner, training=training)
        inner = Dense(250, activation='relu')(inner)
    
    inner = Dropout(0.2)(inner, training=training)
    inner = Dense(200, activation='relu')(inner)
    
    inner = Dropout(0.2)(inner, training=training)
    inner = Dense(100, activation='relu')(inner)
    
    inner = Dropout(0.2)(inner, training=training)
    inner = Dense(20, activation='relu')(inner)
    
    output = Dense(1)(inner)
    
    model_for_pruning = Model(inputs=input0, outputs=output)
    
    # Set up pruning
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Compute end step to finish pruning after 2 epochs.
    num_images = train_images.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    # Define model for pruning.
    pruning_params = {
          'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.00,
                                                                   final_sparsity=0.50,
                                                                   begin_step=0,
                                                                   end_step=end_step)
    }

    model_for_pruning = prune_low_magnitude(model_for_pruning, **pruning_params)

    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(loss=negloglik,optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.))

    # Summary of model used
    print(model_for_pruning.summary())
    
    return model_for_pruning


for fold in range(10):
    # Load weights in model
    model_for_pruning = model()
    model_for_pruning.load_weights(outputdir+"pruning50_CNN_weights_nowedge.h5")

    #save predictions
    print('saving predictions for fold', fold+1, '...')
    eval_data, eval_labels, Ntot = test_images, test_labels, len(test_labels)

    # The Predict() method -  is for the actual prediction. It generates output predictions for the input samples.
    preds = model_for_pruning.predict(eval_data, verbose=0).flatten() #0 = silent

    Nregressparams = len(eval_labels[0])

    results = np.zeros((Ntot, Nregressparams),
                           dtype = [('truth', 'f'), ('prediction', 'f'), ('fold', 'i')])
    iend = istart+len(eval_labels)

    #print('istart and iend', istart, iend)

    results['fold'] = fold
    results['truth'] = eval_labels

    for n in range(Nregressparams):
        results['prediction'][:,n] = preds[n::Nregressparams]
        #results['prediction'][istart-100:iend-100,n] = preds[n::Nregressparams]

    np.save(outputdir+"bestmodel_pred_results_{}".format(str(fold+1)), results)
    
    #destroy the current TF graph and creates a new one
    print("removing model history and TF graph...")
    del model_for_pruning
    gc.collect()
    K.clear_session()
