# needs to be ran in hp_opt environment with Tensorflow version
# https://www.depends-on-the-definition.com/model-uncertainty-in-deep-learning-with-monte-carlo-dropout/
# Data set: MNIST

# tf.__version__ : '2.1.0'

from __future__ import print_function, division, absolute_import

import numpy as np, matplotlib.pyplot as plt, gc, time, h5py, keras
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from datetime import timedelta

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePo
oling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
#from sklearn.metrics import accuracy_score

from matplotlib.ticker import PercentFormatter

gc.enable()

# As you are trying to use function decorator in TF 2.0, please enable run function eagerly by using below line after importing TensorFlow: https://www.tensorflow.org/guide/effective_tf2
#use_tfconfigexperimental_run_functions_eagerly_when_debugging
tf.config.experimental_run_functions_eagerly(True)

wedge = False # Is the data wedge filtered
data_path = '/lustre/aoc/projects/hera/tbilling/ml/data/'

if wedge == False:
    inputFile = data_path+'t21_snapshots_nowedge_v9.hdf5'

if wedge == True:
    inputFile = data_path+'t21_snapshots_wedge_v9.hdf5'

outputdir = "/lustre/aoc/projects/hera/tbilling/ml/baby_densevariational_end_only/"

train_test_file = data_path+"train_test_index_80_20_split.npz"

n=0
N_EPOCH = 2000
factor =1000.


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

# Loss Function
negloglik = lambda y_true, y_pred: -y_pred.log_prob(y_true)

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

        inner = Dropout(0.2)(inner, training=True)
        inner = Dense(350, activation='relu')(inner)

    else:
        inner = Conv2D(128, kernel_size=(3, 3), activation='relu')(inner)
        inner = BatchNormalization()(inner)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)

        inner = Conv2D(128, kernel_size=(3, 3), activation='relu')(inner)
        inner = BatchNormalization()(inner)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)

        inner = GlobalAveragePooling2D()(inner)

        inner = Dropout(0.2)(inner, training=True)
        inner = Dense(250, activation='relu')(inner)

    inner = Dropout(0.2)(inner, training=True)
    inner = Dense(200, activation='relu')(inner)

    inner = Dropout(0.2)(inner, training=True)
    inner = Dense(100, activation='relu')(inner)

    inner = Dropout(0.2)(inner, training=True)
    inner = Dense(20, activation='relu')(inner)

    #output = Dense(1)(inner)
    output1 = tfp.layers.DenseFlipout(1)(inner)#, kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn()
    #kernel_prior_fn=tfp.layers.default_multivariate_normal_fn)(inner)
    output = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1),
             convert_to_tensor_fn=tfp.distributions.Distribution.sample)(output1)

    model_dropout = Model(inputs=input0, outputs=output)

    # Compile Model
    #model_dropout.compile(loss=Mean_Squared_over_true_Error,optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.))
    model_dropout.compile(loss=negloglik,optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.))
    # Summary of model used
    print(model_dropout.summary())

    return model_dropout

model_dropout = model()


filepath_model = outputdir+"denseflip_CNN_model-{epoch:02d}-{loss:.4f}.h5"
filepath_weight = outputdir+"denseflip_CNN_weight-{epoch:02d}-{loss:.4f}.h5"
checkpoint_model = ModelCheckpoint(filepath_model, monitor='loss', verbose=1,
                 save_best_only=True, save_weights_only = False, mode='min', save_freq=200)
checkpoint_weight = ModelCheckpoint(filepath_weight, monitor='loss', verbose=1,
                 save_best_only=True, save_weights_only = True, mode='min', save_freq=200)
callbacks_list = [checkpoint_model, checkpoint_weight]

start_time = time.time()

history_dropout = model_dropout.fit(train_images, train_labels, epochs=N_EPOCH, callbacks=callbacks_list,
                                 batch_size=32, validation_split=0.1, verbose = 2, shuffle=True)

running_time = time.time() - start_time
print("Finish Training CNN in ", str(timedelta(seconds=running_time)))


if wedge == False:
    # Save model
    print("saving model trained on nowedge filtered data ...")
    model_dropout.save(outputdir+"denseflip_at_end_CNN_model_nowedge.h5")
    model_dropout.save_weights(outputdir+"denseflip_at_end_CNN_weights_nowedge.h5")

    # Save history
    print("Removing Scaling factor ({}) and saving histories...".format(factor))
    history_keys = np.array(list(history_dropout.history.keys()))
    for key in history_keys:
        np.savez(outputdir+"denseflip_at_end_CNN_nowedge_history",
                metric=np.array(history_dropout.history[str(key)])/factor)
