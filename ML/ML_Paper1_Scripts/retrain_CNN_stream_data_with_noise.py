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
from keras.callbacks import ModelCheckpoint
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
data_path = "/lustre/aoc/projects/hera/tbilling/ml/data/" #"/pylon5/as5phnp/tbilling/data/"

if wedge == False:
    #inputFile = data_path+'v12_nowedge.h5'
    outputdir = "/lustre/aoc/projects/hera/tbilling/ml/redo_mlpaper/noise/no_modes_removed/" #"/pylon5/as5phnp/tbilling/sandbox/redo_mlpaper/noise/no_modes_removed/"
    noise_list = [data_path+"v12_nowedge_noise_0.1.h5", data_path+"v12_nowedge_noise_0.01.h5", data_path+"v12_nowedge_noise_0.001.h5"]

if wedge == True:
    #inputFile = data_path+'v12_wedge.h5'
    outputdir = "/lustre/aoc/projects/hera/tbilling/ml/redo_mlpaper/noise/modes_removed/" #"/pylon5/as5phnp/tbilling/sandbox/redo_mlpaper/noise/modes_removed/"
    noise_list = [data_path+"v12_wedge_noise_0.1.h5", data_path+"v12_wedge_noise_0.01.h5", data_path+"v12_wedge_noise_0.001.h5"]

n=32
N_EPOCH = 100
factor =1000.
input_shape = (512, 512, 30)

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
    
    model_dropout = Model(inputs=input0, outputs=output)
    
    # Compile Model
    model_dropout.compile(loss=Mean_Squared_over_true_Error,optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.))

    # Summary of model used
    print(model_dropout.summary())
    
    return model_dropout
 
# Generator Function
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format="channels_last",
    validation_split=0.0,
    dtype=None,
)
noise = ["0.1","0.01","0.001"]
for n_idx, inputFile in enumerate(np.array(noise_list)):

    kf = KFold(n_splits=5, random_state=0, shuffle=True)
    # len(train_labels) = 800

    for fold, [train_index, val_index] in enumerate(kf.split(np.arange(800))):
        
        model_dropout = model()
        loss_list = []
        val_loss_list = []
        histories = []
        
        start_time = time.time()
        
        for e in range(N_EPOCH):
            print("*"*50)
            print('Epoch', e+1)
            print("*"*50)
            batches = 1
            
            # Loop through the entire tranining data for ach epoch
            for i in range(0, int(1000*.8), n): #Feed in 21cm data in groups of n=32 at a time
                x_train_ = h5py.File(inputFile, 'r')['train_images'][i:i + n,:,:,:]
                y_train_ = h5py.File(inputFile, 'r')['train_labels'][i:i + n,0]
                for x_batch, y_batch in datagen.flow(x_train_, y_train_, batch_size=32):
                    # break the n samples in to a batch
                    history = model_dropout.fit(x_batch, y_batch, validation_split=0.2, verbose=2) # Train and Validate batches of data aka incremental learning.
                    print("Batch number ",batches)
                    batches += 1
                    if batches >= ((len(x_train_)/32) -1):
                        # Add history
                        print("Adding to histories ... ")
                        loss_list.append(history.history["loss"][0])
                        val_loss_list.append(history.history["val_loss"][0])
                        # we need to break the loop by hand because
                        # the generator loops indefinitely
                        del(y_batch,x_batch,x_train_,y_train_)
                        gc.collect()
                        #K.clear_session()
                        break

        histories.append([loss_list, val_loss_list])
        running_time = time.time() - start_time
        print("Finish Training CNN in ", str(timedelta(seconds=running_time)))
        
        # Save predictions
        test_images = h5py.File(inputFile, 'r')['test_images']
        test_labels = h5py.File(inputFile, 'r')['test_labels']
        savePreds(model_dropout, test_images, test_labels, len(test_labels), fold)
                
        # Save model information
        if wedge == True:
            # Save model
            print("saving model trained on wedge filtered data ...")
            model_dropout.save_weights(outputdir+"CNN_weights_wedge_{}_{}.h5".format(fold, noise[n_idx]))
            model_dropout.save(outputdir+"CNN_model_wedge_{}_{}.h5".format(fold, noise[n_idx]))
            
            # Save history
            print("Removing Scaling factor ({}) and saving histories...".format(factor))
            #history_keys = np.array(list(history_dropout.history.keys()))
            np.savez(outputdir+"CNN_wedge_history_{}_{}".format(fold, noise[n_idx]), metric = np.array(histories))#,loss=np.array(history_dropout.history[str("loss")])/factor, val_loss=np.array(history_dropout.history[str("val_loss")])/factor)
                         
        if wedge == False:
            # Save model
            print("saving model trained on nowedge filtered data ...")
            model_dropout.save_weights(outputdir+"CNN_weights_nowedge_{}_{}.h5".format(fold, noise[n_idx]))
            model_dropout.save(outputdir+"CNN_model_nowedge_{}_{}.h5".format(fold, noise[n_idx]))

            # Save history
            print("Removing Scaling factor ({}) and saving histories...".format(factor))
            #history_keys = np.array(list(history_dropout.history.keys()))
            np.savez(outputdir+"CNN_nowedge_history_{}_{}".format(fold, noise[n_idx]), metric = np.array(histories))#,loss=np.array(history_dropout.history[str("loss")])/factor, val_loss=np.array(history_dropout.history[str("val_loss")])/factor)
            
        # remove TF graph
        del(model_dropout, test_images, test_labels)
        gc.collect()
        K.clear_session()
        #tf.compat.v1.reset_default_graph()
"""
for i in range(10):
    indx_val = Rand(0, 199, 100)
    savePreds(model_dropout, test_images[indx_val], test_labels[indx_val], len(test_labels[indx_val]), fold=i+1,  outdir=outputdir)
    print("Completed Fold ", i)
"""
