from __future__ import print_function, division, absolute_import

import numpy as np, matplotlib.pyplot as plt, gc, time, h5py, keras
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from datetime import timedelta

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
#from sklearn.metrics import accuracy_score

from matplotlib.ticker import PercentFormatter

import gc
gc.enable()


wedge = False # Is the data wedge filtered
data_path = "/lustre/aoc/projects/hera/tbilling/ml/data/" #"/pylon5/as5phnp/tbilling/data/"

if wedge == False:
    inputFile = data_path+"t21_snapshots_nowedge_v12.hdf5"

if wedge == True:
    inputFile = data_path+"t21_snapshots_wedge_v12.hdf5"

outputdir = "/lustre/aoc/projects/hera/tbilling/ml/baby_densevariational_end_only/" #"/pylon5/as5phnp/tbilling/sandbox/bayesian/denseflipout/"
train_test_file = data_path+"train_test_index_80_20_split.npz"

# Potential Parameters to change
l1 = 1e-4
l2 = 1e-3
lr = 1e-3
batch_size = int(32*1.5)#32

n = batch_size
N_EPOCH = 400#2000
factor =1000.
input_shape = (512, 512, 30)
training = False

# Loss Function
negloglik = lambda y_true, y_pred: -y_pred.log_prob(y_true)

# Model
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
        #inner = Dense(350, activation='relu',
        #        kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
        #        bias_regularizer=regularizers.l2(l2),
        #        activity_regularizer=regularizers.l2(1e-5)
        #        )(inner)

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
        #inner = Dense(250, activation='relu',
        #        kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
        #        bias_regularizer=regularizers.l2(l2),
        #        activity_regularizer=regularizers.l2(1e-5)
        #        )(inner)

    inner = Dropout(0.2)(inner, training=training)
    inner = Dense(200, activation='relu')(inner)
    #inner = Dense(200, activation='relu',
    #        kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
    #        bias_regularizer=regularizers.l2(l2),
    #        activity_regularizer=regularizers.l2(1e-5)
    #        )(inner)

    inner = Dropout(0.2)(inner, training=training)
    inner = Dense(100, activation='relu')(inner)
    #inner = Dense(100, activation='relu',
    #        kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
    #        bias_regularizer=regularizers.l2(l2),
    #        activity_regularizer=regularizers.l2(1e-5)
    #        )(inner)

    inner = Dropout(0.2)(inner, training=training)
    inner = Dense(20, activation='relu')(inner)
    #inner = Dense(20, activation='relu',
    #        kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
    #        bias_regularizer=regularizers.l2(l2),
    #        activity_regularizer=regularizers.l2(1e-5)
    #        )(inner)

    #output = Dense(1)(inner)
    output1 = tfp.layers.DenseFlipout(1)(inner)#, kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
                #kernel_prior_fn=tfp.layers.default_multivariate_normal_fn)(inner)
    output = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1),
                convert_to_tensor_fn=tfp.distributions.Distribution.sample)(output1)

    model_dropout = Model(inputs=input0, outputs=output)

    # Compile Model
    #model_dropout.compile(loss=Mean_Squared_over_true_Error,optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.))

    model_dropout.compile(loss=negloglik,optimizer=keras.optimizers.Adam(lr=lr, decay=0.))
    # Summary of model used
    print(model_dropout.summary())

    return model_dropout

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

print("Building Model...")
loss_list = []
val_loss_list = []
histories = []
model_dense = model()

# here's a more "manual" example
for e in range(N_EPOCH):
    print("*"*50)
    print('Epoch', e+1)
    print("*"*50)
    batches = 1

    # Save a copy of the weight/model every 100 epoch
    if e+1 in range(100,N_EPOCH+100, 100):

        filepath_model = outputdir+"denseflip_CNN_model-epoch{}.h5".format(str(e+1))
        filepath_weight = outputdir+"denseflip_CNN_weight-epoch{}.h5".format(str(e+1))
        checkpoint_model = ModelCheckpoint(filepath_model, monitor='loss', verbose=1,
                            save_best_only=True, save_weights_only = False, mode='min', save_freq=1)
        checkpoint_weight = ModelCheckpoint(filepath_weight, monitor='loss', verbose=1,
                            save_best_only=True, save_weights_only = True, mode='min', save_freq=1)
        callbacks_list = [checkpoint_model, checkpoint_weight]

        print("Trying to save weights and model for epoch {}".format(str(e+1)))
        for i in range(0, int(1000*.75), batch_size): #Feed in 21cm data in groups of n=32 at a time
            x_train_ = ((np.asarray(h5py.File(inputFile, 'r')['Data'][u't21_snapshots']).transpose(0,2,3,1))[i:i + n,:,:,:])
            y_train_ = np.asarray(h5py.File(inputFile, 'r')['Data'][u'snapshot_labels'][i:i + n,5]).reshape(-1, 1)*factor
            for x_batch, y_batch in datagen.flow(x_train_, y_train_, batch_size=batch_size):
                history = model_dense.fit(x_batch, y_batch, validation_split=0.2, callbacks=callbacks_list, verbose=1)
                print("Batch number ",batches)
                batches += 1
                if batches >= ((len(x_train_)/batch_size) -1):
                    # Add history
                    print("Adding to histories ... ")
                    loss_list.append(history.history["loss"][0])
                    val_loss_list.append(history.history["val_loss"][0])
                    
                    del(y_batch,x_batch,x_train_,y_train_)
                    gc.collect()
                    K.clear_session()
                    break
            break

    for i in range(0, int(1000*.75), batch_size): #Feed in 21cm data in groups of n=32 at a time
        x_train_ = ((np.asarray(h5py.File(inputFile, 'r')['Data'][u't21_snapshots']).transpose(0,2,3,1))[i:i + n,:,:,:])
        y_train_ = np.asarray(h5py.File(inputFile, 'r')['Data'][u'snapshot_labels'][i:i + n,5]).reshape(-1, 1)*factor
        for x_batch, y_batch in datagen.flow(x_train_, y_train_, batch_size=batch_size):
            # break the n samples in to a batch
            history = model_dense.fit(x_batch, y_batch, validation_split=0.2, verbose=1) # Train and Validate batches of data aka incremental learning.
            print("Batch number ",batches)
            batches += 1
            if batches >= ((len(x_train_)/batch_size) -1):
                # Add history
                print("Adding to histories ... ")
                loss_list.append(history.history["loss"][0])
                val_loss_list.append(history.history["val_loss"][0])
                
                # we need to break the loop by hand because
                # the generator loops indefinitely
                del(y_batch,x_batch,x_train_,y_train_)
                gc.collect()
                K.clear_session()
                break

# Save history
histories.append([loss_list, val_loss_list])


if wedge == False:
    print("Ssaving histories...")
    np.savez(outputdir+"denseflipout_CNN_nowedge_history",metric = np.array(histories),
    loss = np.array(loss_list), val_loss = np.array(val_loss_list))

    print("saving model trained on nowedge filtered data ...")
    model_dense.save_weights(outputdir+"denseflipout_CNN_weights_nowedge.h5")
    model_dense.save(outputdir+"denseflipout_CNN_model_nowedge.h5")

if wedge == True:
    print("Ssaving histories...")
    np.savez(outputdir+"denseflipout_CNN_wedge_history",metric = np.array(histories),
    loss = np.array(loss_list), val_loss = np.array(val_loss_list))

    print("saving model trained on nowedge filtered data ...")
    model_dense.save_weights(outputdir+"denseflipout_CNN_weights_wedge.h5")
    model_dense.save(outputdir+"denseflipout_CNN_model_wedge.h5")

