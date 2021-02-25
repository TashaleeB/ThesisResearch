# needs to be ran in hp_opt environment with Tensorflow version
# https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras
# https://www.tensorflow.org/model_optimization/guide/pruning/comprehensive_guide
# Data set: v9data

# tf.__version__ : ''

from __future__ import print_function, division, absolute_import

import numpy as np, matplotlib.pyplot as plt, gc, time, h5py, keras, tempfile
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_model_optimization as tfmot

tfd = tfp.distributions

from datetime import timedelta

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
#from sklearn.metrics import accuracy_score

from matplotlib.ticker import PercentFormatter

gc.enable()

# As you are trying to use function decorator in TF 2.0, please enable run function eagerly by using below line after importing TensorFlow: https://www.tensorflow.org/guide/effective_tf2#use_tfconfigexperimental_run_functions_eagerly_when_debugging
tf.config.experimental_run_functions_eagerly(True)

wedge = False # Is the data wedge filtered
data_path = '/pylon5/as5phnp/tbilling/data/'

if wedge == False:
    inputFile = data_path+'t21_snapshots_nowedge_v9.hdf5'

if wedge == True:
    inputFile = data_path+'t21_snapshots_wedge_v9.hdf5'
    
best_model_name = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/opt_versions/200steps_v9/hyperParam_model_1.h5"
outputdir = "/pylon5/as5phnp/tbilling/sandbox/pruning/"

train_test_file = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/train_test_index_0.npz"

n=0
epochs = 200
batch_size = 32
validation_split = 0.1 # 10% of training set will be used for validation set.
factor =1000.
training = False # Set to True if dropout active in inference mode.


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

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# Load Index Label
train_index = np.load(train_test_file)["train_index"]
test_index = np.load(train_test_file)["test_index"]

# Load images and labels for training and testing
train_labels = readLabels(ind=None)[train_index,5]*factor
train_labels = train_labels.reshape(-1, 1)
train_images,shape =readImages(ind=train_index)

test_labels = readLabels(ind=None)[test_index,5]*factor
testl_abels = test_labels.reshape(-1, 1)
test_images,input_shape = readImages(ind=test_index)

"""
# It's generally better to finetune with pruning as opposed to training from scratch.
# Load the best model
model_for_pruning = load_model(best_model_name,
custom_objects={"Mean_Squared_over_true_Error": Mean_Squared_over_true_Error,
"r2_keras": r2_keras})

model_for_pruning.load_weights("pruning50_CNN_weights_nowedge.h5")
print(model_for_pruning.summary())
"""

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
    
    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(loss=Mean_Squared_over_true_Error,optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.))
    
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

    # Summary of model used
    print(model_for_pruning.summary())
    
    return model_for_pruning

model_for_pruning = model()
logdir = tempfile.mkdtemp()

filepath_model = outputdir+"prune_CNN_model.h5"
filepath_weight = outputdir+"prune_CNN_weight.h5"
checkpoint_model = ModelCheckpoint(filepath_model, monitor='loss', verbose=1,
                                   save_best_only=True, save_weights_only = False, mode='min', save_freq=20)
checkpoint_weight = ModelCheckpoint(filepath_weight, monitor='loss', verbose=1,
                                    save_best_only=True, save_weights_only = True, mode='min', save_freq=20)
callbacks = [checkpoint_model, checkpoint_weight,
    tfmot.sparsity.keras.UpdatePruningStep(),
    # Log sparsity and other metrics in Tensorboard.
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir, update_freq='epoch')]


start_time = time.time()
# Start Training
history_pruning = model_for_pruning.fit(train_images, train_labels, epochs=epochs,callbacks=callbacks,
                                    batch_size=batch_size, validation_split=validation_split, verbose = 2, shuffle=True)

running_time = time.time() - start_time
print("Finish Training CNN in ", str(timedelta(seconds=running_time)))
        
# Save model information
if wedge == True:
    # Save model and weights
    print("saving model trained on wedge filtered data ...")
    model_for_pruning.save(outputdir+"pruning_CNN_model_wedge.h5")
    model_for_pruning.save_weights(outputdir+"pruning_CNN_weights_wedge.h5")

    # Save history
    print("Removing Scaling factor ({}) and saving histories...".format(factor))
    history_keys = np.array(list(history_pruning.history.keys()))
    np.savez(outputdir+"pruning_CNN_wedge_history",loss=np.array(history_pruning.history[str("loss")])/factor, val_loss=np.array(history_pruning.history[str("val_loss")])/factor)
    
 """
history_keys = np.array(list(history_dropout.history.keys()))
np.savez(outputdir+"transfer_model_CNN_nowedge_history",loss=np.array(history_dropout.history[str("loss")])/factor, val_loss=np.array(history_dropout.history[str("val_loss")])/factor)
"""
if wedge == False:
    # Save model and weights
    print("saving model trained on nowedge filtered data ...")
    model_for_pruning.save(outputdir+"pruning_CNN_model_nowedge.h5")
    model_for_pruning.save_weights(outputdir+"pruning_CNN_weights_nowedge.h5")

    # Save history
    print("Removing Scaling factor ({}) and saving histories...".format(factor))
    history_keys = np.array(list(history_pruning.history.keys()))
    np.savez(outputdir+"pruning_CNN_nowedge_history",loss=np.array(history_pruning.history[str("loss")])/factor, val_loss=np.array(history_pruning.history[str("val_loss")])/factor)

# evaluate trained model
test_loss = model_for_pruning.evaluate(test_images, test_labels)
