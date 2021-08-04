from __future__ import print_function, division, absolute_import

import numpy as np, matplotlib.pyplot as plt, datetime, time, h5py, keras, random, os
import tensorflow as tf

from datetime import timedelta

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split, KFold
#from sklearn.metrics import accuracy_score

#from matplotlib.ticker import PercentFormatter

import gc
gc.enable()

# As you are trying to use function decorator in TF 2.0, please enable run function eagerly by using below line after importing TensorFlow: https://www.tensorflow.org/guide/effective_tf2#use_tfconfigexperimental_run_functions_eagerly_when_debugging
tf.config.experimental_run_functions_eagerly(True)

wedge = False # Is the data wedge filtered
training = False # if True the dropout will be active during to testing processes
data_path = "/ocean/projects/ast180004p/tbilling/data/"

if wedge == False:
    inputFile = data_path+"t21_snapshots_nowedge_v12.hdf5"
    

if wedge == True:
    inputFile = data_path+"t21_snapshots_wedge_v12.hdf5"
    
outputdir = "/ocean/projects/ast180004p/tbilling/sandbox/bayesian/denseflipout/sandbox/tensorboard/"

fold = 0
BATCH=32
N_EPOCH = 100
factor =1000.
input_shape = (512, 512, 30)

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
    
    model = Model(inputs=input0, outputs=output)
    
    # Compile Model
    model.compile(loss=Mean_Squared_over_true_Error,optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.))

    # Summary of model used
    print(model.summary())
    
    return model

# Load training data
training_labels = readLabels(ind=None)[np.arange(800),5]*factor
training_labels = training_labels.reshape(-1, 1)
training_images,input_shape = readImages(ind=np.arange(800))

# Load test data
test_labels = readLabels(ind=None)[np.arange(800,1000),5]*factor
test_labels = test_labels.reshape(-1, 1)
test_images,input_shape = readImages(ind=np.arange(800,1000))

# logs in a timestamped subdirectory
log_dir = outputdir + "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# Initialize Model
model = model()
history = model.fit(training_images, training_labels, validation_split=0.2, verbose=2,
                    batch_size=BATCH, epochs=N_EPOCH, callbacks=[tensorboard_callback]) # Train and Validate batches of data aka incremental learning.


if fold == 0:
    if wedge == True:
        # Save model
        print("saving model trained on wedge filtered data ...")
        model.save_weights(outputdir+"CNN_weights_wedge_{}.h5".format(fold))
        model.save(outputdir+"CNN_model_wedge_{}.h5".format(fold))
        
        # Save history
        print("Removing Scaling factor ({}) and saving histories...".format(factor))

        np.savez(outputdir+"CNN_wedge_history_{}".format(fold), metric = history)
                     
    if wedge == False:
        # Save model
        print("saving model trained on nowedge filtered data ...")
        model.save_weights(outputdir+"CNN_weights_nowedge_{}.h5".format(fold))
        model.save(outputdir+"CNN_model_nowedge_{}.h5".format(fold))

        # Save history
        print("Removing Scaling factor ({}) and saving histories...".format(factor))

        np.savez(outputdir+"CNN_nowedge_history_{}".format(fold), metric = history)
        
# Save predictions
savePreds(model, test_images, test_labels, len(test_labels), 1)

