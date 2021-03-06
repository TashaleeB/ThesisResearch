# needs to be ran in hp_opt environment with Tensorflow version

# Data set: MNIST

# tf.__version__ : '2.1.0'

from __future__ import print_function, division, absolute_import

import numpy as np, matplotlib.pyplot as plt, time, h5py, keras, random, os
import tensorflow as tf

from datetime import timedelta

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split, KFold
#from sklearn.metrics import accuracy_score

#from matplotlib.ticker import PercentFormatter

import gc
gc.enable()

# As you are trying to use function decorator in TF 2.0, please enable run function eagerly by using below line after importing TensorFlow: https://www.tensorflow.org/guide/effective_tf2#use_tfconfigexperimental_run_functions_eagerly_when_debugging
tf.config.experimental_run_functions_eagerly(True)

wedge = False # Is the data wedge filtered
training = False # if True the dropout will be active during to testing processes
data_path = "/lustre/aoc/projects/hera/tbilling/ml/data/" #"/pylon5/as5phnp/tbilling/data/"

if wedge == False:
    inputFile = data_path+'t21_snapshots_nowedge_v12.hdf5'
    outputdir = "/lustre/aoc/projects/hera/tbilling/ml/redo_mlpaper/no_modes_removed/" #"/pylon5/as5phnp/tbilling/sandbox/redo_mlpaper/no_modes_removed/"

if wedge == True:
    inputFile = data_path+'t21_snapshots_wedge_v12.hdf5'
    outputdir = "/lustre/aoc/projects/hera/tbilling/ml/redo_mlpaper/modes_removed/" #"/pylon5/as5phnp/tbilling/sandbox/redo_mlpaper/no_modes_removed/"
    

train_test_file =data_path+"train_test_index_80_20_split.npz"

n=0
N_EPOCH = 100
factor =1000.

def Rand(start, end, num):
    res = []

    for j in range(num):
        res.append(random.randint(start, end))

    return np.array(res)

def readLabels(ind=None, **params):

    # read in labels only

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
    # read in images onle

    print('reading data from', inputFile)

    f = h5py.File(inputFile, 'r')

    #if params['debug'] == True:
    #    data = np.asarray(f['Data'][u't21_snapshots'][ind,:,0:16,0:16])
    if 'crop' in params:
        print('cropping.')
        #use just the top corner of the images
        data = np.asarray(f['Data'][u't21_snapshots'][ind,:,0:params['crop'],0:params['crop']])
    else:
        #use everything!
        print('reading all data', len(ind))
        data = np.asarray(f['Data'][u't21_snapshots'][ind,:,:,:]) # (N_realizations, N_redshifts, N_pix, N_pix)
        print('loaded data', len(ind))

    print('finished loading data.', data.shape)

    data  = data.transpose(0,2,3,1) #(N_realizations, N_pix, N_pix, N_redshifts)

    return data, data[0].shape

def Mean_Squared_over_true_Error(y_true, y_pred):
    # Create a custom loss function that divides the difference by the true
    #if not K.is_tensor(y_pred):
    #if not K.is_keras_tensor(y_pred):
    #    y_pred = K.constant(y_pred)

    y_true = K.cast(y_true, y_pred.dtype) #Casts a tensor to a different dtype and returns it.
    diff_ratio = K.square((y_pred - y_true)/K.clip(K.abs(y_true),K.epsilon(),None))
    loss = K.mean(diff_ratio, axis=-1)
    # Return a function

    return loss

def savePreds(model, eval_data, eval_labels, Ntot, fold, istart=0, outdir=outputdir):
    outputFile = os.path.join(outdir, "eval_pred_results{:d}_v12data.npy".format(fold))

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

"""
# Load Index Label
train_index = np.load(train_test_file)["train_index"]
test_index = np.load(train_test_file)["test_index"]

# Load images and labels for training and testing
train_labels = readLabels(ind=None)[train_index,5]*factor
train_labels = train_labels.reshape(-1, 1)
train_images,shape =readImages(ind=train_index)

test_labels = readLabels(ind=None)[test_index,5]*factor
test_labels = test_labels.reshape(-1, 1)
test_images,input_shape = readImages(ind=test_index)
"""

# Train/Valdiation/Test Data
y = readLabels(ind=None)[:,5]*factor
y = y.reshape(-1, 1)
X,input_shape =readImages(ind=np.arange(1000))

train_images, test_images, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=0)

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
    
    model_dropout = Model(inputs=input0, outputs=output)
    
    # Compile Model
    model_dropout.compile(loss=Mean_Squared_over_true_Error,optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.))

    # Summary of model used
    print(model_dropout.summary())
    
    return model_dropout
    

start_time = time.time()

kf = KFold(n_splits=10, random_state=0, shuffle=True)
for fold, [train_index, val_index] in enumerate(kf.split(np.arange(len(train_labels)))):

    # Start Training
    model_dropout = model()
    history_dropout = model_dropout.fit(train_images, train_labels, epochs=N_EPOCH, batch_size=32, validation_data=(train_images[val_index], train_labels[val_index]), verbose = 2, shuffle=True)
    #history_dropout = model_dropout.fit(train_images, train_labels, epochs=N_EPOCH,
    #                                    batch_size=32, validation_split=0.1, verbose = 2, shuffle=True)

    running_time = time.time() - start_time
    print("Finish Training CNN in ", str(timedelta(seconds=running_time)))
    
    # Save predictions
    savePreds(model_dropout, test_images, test_labels, len(test_labels), fold)
            
    # Save model information
    if wedge == True:
        # Save model
        print("saving model trained on wedge filtered data ...")
        model_dropout.save_weights(outputdir+"CNN_wights_wedge_{}.h5".format(fold))
        model_dropout.save(outputdir+"CNN_model_wedge_{}.h5".format(fold))
        
        # Save history
        print("Removing Scaling factor ({}) and saving histories...".format(factor))
        #history_keys = np.array(list(history_dropout.history.keys()))
        np.savez(outputdir+"CNN_wedge_history_{}".format(fold),loss=np.array(history_dropout.history[str("loss")])/factor, val_loss=np.array(history_dropout.history[str("val_loss")])/factor)
                     
    if wedge == False:
        # Save model
        print("saving model trained on nowedge filtered data ...")
        model_dropout.save_weights(outputdir+"CNN_weights_nowedge_{}.h5".format(fold))
        model_dropout.save(outputdir+"CNN_model_nowedge_{}.h5".format(fold))

        # Save history
        print("Removing Scaling factor ({}) and saving histories...".format(factor))
        #history_keys = np.array(list(history_dropout.history.keys()))
        np.savez(outputdir+"CNN_nowedge_history_{}".format(fold),loss=np.array(history_dropout.history[str("loss")])/factor, val_loss=np.array(history_dropout.history[str("val_loss")])/factor)
        
    # remove TF graph
    del(model_dropout)
    gc.collect()
    K.clear_session()
    #tf.compat.v1.reset_default_graph()

for i in range(10):
    indx_val = Rand(0, 199, 100)
    savePreds(model_dropout, test_images[indx_val], test_labels[indx_val], len(test_labels[indx_val]), fold=i+1,  outdir=outputdir)
    print("Completed Fold ", i)
