"""
https://stackoverflow.com/questions/41668813/how-to-add-and-remove-new-layers-in-keras-after-loading-weights
"""
from __future__ import print_function, division, absolute_import

import numpy as np, matplotlib.pyplot as plt, gc, time, h5py, keras
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from datetime import timedelta

from tensorflow.keras import Model
from tensorflow.keras.models import load_model

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras import regularizers

from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


from matplotlib.ticker import PercentFormatter

import gc
gc.enable()


wedge = False # Is the data wedge filtered
data_path = "/ocean/projects/ast180004p/tbilling/data/"

if wedge == False:
    inputFile = data_path+"t21_snapshots_nowedge_v12.hdf5"
    perfectmodel = "/ocean/projects/ast180004p/tbilling/sandbox/redo_mlpaper/no_modes_removed/CNN_model_nowedge_1.h5" # nowedge

if wedge == True:
    inputFile = data_path+"t21_snapshots_wedge_v12.hdf5"
    perfectmodel = "/ocean/projects/ast180004p/tbilling/sandbox/redo_mlpaper/modes_removed/CNN_model_wedge_5.h5" # wedge

outputdir = "/ocean/projects/ast180004p/tbilling/sandbox/bayesian/denseflipout/sandbox/load_ml_p1_model_stream_v12Data_DenseVar_layer_end/"

# Potential Parameters to change
batch_size = 32

n = batch_size
N_EPOCH = 400
factor =1000.
#input_shape = (512, 512, 30)

lr = 1e-3


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

# Load training data
training_labels = readLabels(ind=None)[np.arange(800),5]*factor
training_labels = training_labels.reshape(-1, 1)
training_images,input_shape = readImages(ind=np.arange(800))
num_examples = training_labels.shape[0]

# Load Model
model = load_model(perfectmodel,
custom_objects={"Mean_Squared_over_true_Error": Mean_Squared_over_true_Error})
print(model.summary())

# Make layer not trainable
print("Making layers not trainable")
for l in model.layers:
    l.trainable = False
    print(l.trainable)

print("Editing the last layer")
#editing layers in the second model and saving as third model
output = tfp.layers.DenseFlipout(1)(model.layers[-2].output)#, kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),

model_DenseFlipout = Model(inputs=model.input, outputs=[output])
model_DenseFlipout.summary()
   
print("Building Model...")
loss_list = []
val_loss_list = []
histories = []

model_DenseFlipout.load_weights("denseflip_CNN_weight_wedge-epoch391.h5") # continue training if time runs out
model_DenseFlipout.compile(loss=Mean_Squared_over_true_Error,optimizer=keras.optimizers.Adam(lr=lr, decay=0.))



"""
# Load Model
model = load_model(perfectmodel,
custom_objects={"Mean_Squared_over_true_Error": Mean_Squared_over_true_Error})
print(model.summary())

#loading weights to second model
model2= model
#model2.layers.pop()
#model2.layers.pop()
model2.summary()

#editing layers in the second model and saving as third model
output1 = tfp.layers.DenseFlipout(1)(model2.layers[-2].output)#, kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
            #kernel_prior_fn=tfp.layers.default_multivariate_normal_fn)(inner)
output = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1),
            convert_to_tensor_fn=tfp.distributions.Distribution.sample)(output1)
model_DenseFlipout = Model(inputs=model.input, outputs=[output])
model_DenseFlipout.summary()

# complile model
#model_DenseFlipout.compile(loss=negloglik,optimizer=keras.optimizers.Adam(lr=lr, decay=0.))
model_DenseFlipout.compile(loss=tf.keras.losses.KLDivergence(),optimizer=keras.optimizers.Adam(lr=lr, decay=0.))

#history_bnn_model = model_DenseFlipout.fit(training_images[:num_examples,:,:,:],
#                                    training_labels[:num_examples], epochs=N_EPOCH,
#                                    batch_size=batch_size, validation_split=0.1, verbose = 2, shuffle=True)

"""

"""
# Loss Function
#negloglik = lambda y_true, y_pred: -y_pred.log_prob(y_true)
#neg_log_likelihood = lambda y_true, y_pred: -tf.reduce_mean(input_tensor=y_pred.log_prob(y_true))

# Loss functions applied to the output of a model aren't the only way to create losses.
# Compute scalar quantities that you want to minimize during training (e.g. regularization losses). The add_loss() layer method to keep track of such loss terms.
model_DenseFlipout.add_loss(lambda: tf.reduce_mean(model_DenseFlipout.layers[1].kernel)) # create a regularization loss
    # that depends on the inputs.
losses_values = model_DenseFlipout.losses # Loss values added via add_loss can be retrieved in the .losses. Compute the quantity that a model should seek to minimize during training.

neg_log_likelihood = -tf.reduce_mean(input_tensor=model_DenseFlipout(training_images[0:1]).log_prob(training_labels))
kl = tf.keras.losses.KLDivergence()/num_examples #sum(losses_values) / num_examples
elbo_loss = neg_log_likelihood + kl


def elbo(y_true, y_pred):
    kl_weight = 1
    neg_log_likelihood = lambda y_true, y_pred: -tf.reduce_mean(input_tensor=y_pred.log_prob(y_true))
    kl_divergence = tf.keras.losses.KLDivergence()#/800 # [kldiv_function / num_examples]
    
    elbo_loss = -tf.math.reduce_mean(-kl_weight * kl_divergence - neg_log_likelihood)
    return elbo_loss


"""


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


# here's a more "manual" example
for e in range(N_EPOCH):
    print("*"*50)
    print('Epoch', e+1)
    print("*"*50)
    batches = 1

    # Save a copy of the weight/model every n number of epoch
    if e in np.linspace(start=0,stop=N_EPOCH, num=41, endpoint=True):
        if wedge == False:
            name = "nowedge"
        if wedge == True:
            name = "wedge"
            
        #e = int(e)
            
        filepath_model = outputdir+"denseflip_CNN_model_{}-epoch{}.h5".format(name, str(e+1))
        filepath_weight = outputdir+"denseflip_CNN_weight_{}-epoch{}.h5".format(name, str(e+1))
        checkpoint_model = ModelCheckpoint(filepath_model, monitor='loss', verbose=1,
                            save_best_only=True, save_weights_only = False, mode='min', save_freq=1)
        checkpoint_weight = ModelCheckpoint(filepath_weight, monitor='loss', verbose=1,
                            save_best_only=True, save_weights_only = True, mode='min', save_freq=1)
        callbacks_list = [checkpoint_model, checkpoint_weight]

        print("Trying to save weights and model for epoch {}".format(str(e+1)))
        for i in range(0, num_examples, batch_size): #Feed in 21cm data in groups of n=32 at a time
            x_train_ = ((np.asarray(h5py.File(inputFile, 'r')['Data'][u't21_snapshots']).transpose(0,2,3,1))[i:i + n,:,:,:])
            y_train_ = np.asarray(h5py.File(inputFile, 'r')['Data'][u'snapshot_labels'][i:i + n,5]).reshape(-1, 1)*factor
            for x_batch, y_batch in datagen.flow(x_train_, y_train_, batch_size=batch_size):
                history = model_DenseFlipout.fit(x_batch, y_batch, validation_split=0.2,
                                    callbacks=callbacks_list, verbose=2, shuffle=True)
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

    for i in range(0, num_examples, batch_size): #Feed in 21cm data in groups of n=32 at a time
        x_train_ = ((np.asarray(h5py.File(inputFile, 'r')['Data'][u't21_snapshots']).transpose(0,2,3,1))[i:i + n,:,:,:])
        y_train_ = np.asarray(h5py.File(inputFile, 'r')['Data'][u'snapshot_labels'][i:i + n,5]).reshape(-1, 1)*factor
        for x_batch, y_batch in datagen.flow(x_train_, y_train_, batch_size=batch_size):
            # break the n samples in to a batch
            history = model_DenseFlipout.fit(x_batch, y_batch, validation_split=0.2, verbose=2, shuffle=True) # Train and Validate batches of data aka incremental learning.
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

