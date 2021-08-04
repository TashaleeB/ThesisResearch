# needs to be ran in dl-gpu environment with Tensorflow version
# tf.__version__ : '2.+'
# https://github.com/tensorflow/probability/issues/511

import numpy as np, matplotlib.pyplot as plt, seaborn as sns
import h5py, progressbar
import tensorflow as tf, tensorflow_probability as tfp

from tensorflow import keras
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

tfd = tfp.distributions

n_pix = 28
factor =1000.
NUM_TRAIN_EXAMPLES = 60000

wedge = False # Is the data wedge filtered
data_path = '/ocean/projects/ast180004p/tbilling/data/'

if wedge == False:
    inputFile = data_path+'t21_snapshots_nowedge_v12.hdf5'

if wedge == True:
    inputFile = data_path+'t21_snapshots_wedge_v12.hdf5'

def readLabels(ind=None, **params):
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

labels = readLabels(ind=None)[np.arange(100),5]*factor
labels = labels.reshape(-1, 1)

images,input_shape = readImages(ind=np.arange(100))
input_shape = (n_pix, n_pix, 30)
images = images[:,0:n_pix,0:n_pix,:]

# Bayesian Neural Net
kl_divergence_function=(lambda q, p, _: tfd.kl_divergence(q,p)/tf.cast(NUM_TRAIN_EXAMPLES, dtype = tf.float32))

bnn_model=models.Sequential()

bnn_model.add(layers.Input(shape=input_shape, name = "input_layer"))

bnn_model.add(tfp.layers.Convolution2DFlipout(32, kernel_size=(3, 3), activation='relu',
name = "hidden_layer_Convflipout_1"))
bnn_model.add(layers.BatchNormalization(name = "normalization"))
bnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name = "pooling"))

bnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name = "max_pooling"))
bnn_model.add(layers.GlobalAveragePooling2D(name = "average_pooling"))

bnn_model.add(tfp.layers.DenseFlipout(100, activation = tf.nn.relu, kernel_divergence_fn = kl_divergence_function, name = "hidden_layer_Denseflipout_1"))
bnn_model.add(tfp.layers.DenseFlipout(20, kernel_divergence_fn = kl_divergence_function, activation = tf.nn.relu, name = "hidden_layer_Denseflipout_2"))
bnn_model.add(tfp.layers.DenseFlipout(1, kernel_divergence_fn = kl_divergence_function, name = "output_layer_Denseflipout"))
    
bnn_model.compile(optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.),
               loss=Mean_Squared_over_true_Error)
print(bnn_model.summary())

# Traing BNN
history_bnn_model = bnn_model.fit(images[:80,:,:,:], labels[:80], epochs=10,
                                batch_size=32, validation_split=0.1, verbose = 2, shuffle=True)


n_bootstrap=1000
n_classes=10 # one for each category

# make array to save predictions
predicted_values=np.empty((n_bootstrap, n_classes), dtype=np.float64)

for i in progressbar.progressbar(range(n_bootstrap)):
    pred_val = bnn_model.predict(test_image[np.newaxis, :, :])
    predicted_values[i, :]=pred_val[0, :]











kl_divergence_function=(lambda q, p, _: tfd.kl_divergence(q, p)/tf.cast(NUM_TRAIN_EXAMPLES,dtype=tf.float32))
    ...:
    ...: #del(bnn_model)
    ...: """
    ...: bnn_model=models.Sequential()
    ...:
    ...: bnn_model.add(layers.Input(shape=(28, 28, 30), name = "input_layer"))
    ...:
    ...: bnn_model.add(tfp.layers.Convolution2DFlipout(32, kernel_size=(3, 3), activation='relu',
    ...: name = "hidden_layer_Convflipout_1"))
    ...: bnn_model.add(layers.BatchNormalization(name = "normalization"))
    ...: bnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name = "pooling"))
    ...:
    ...: bnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name = "max_pooling"))
    ...: bnn_model.add(layers.GlobalAveragePooling2D(name = "average_pooling"))
    ...:
    ...: #bnn_model.add(layers.Flatten())
    ...:
    ...: bnn_model.add(tfp.layers.DenseFlipout(100, activation = tf.nn.relu, kernel_divergence_fn = kl_divergence_function, name = "hidden_layer_Denseflipout_1"))
    ...: bnn_model.add(tfp.layers.DenseFlipout(20, kernel_divergence_fn = kl_divergence_function, activation = tf.nn.relu, name = "hidden_layer_Denseflipout_2"))
    ...: bnn_model.add(tfp.layers.DenseFlipout(1,kernel_divergence_fn=kl_divergence_function,activation=tf.nn.relu,name="output_layer_flipout",))
    ...:
    ...: bnn_model.compile(optimizer="adam", loss="mse")
    ...: print(bnn_model.summary())"""
    ...: history_bnn_model = bnn_model.fit(images[:80,:,:,:], labels[:80], epochs=1000,
    ...:                                 batch_size=32, validation_split=0.1, verbose = 2, shuffle=True)



















# '%0.17f' % np.sum(bnn_model.predict(test_image[np.newaxis, :, :]))
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

kl_divergence_function=(lambda q, p, _: tfd.kl_divergence(q, p)/tf.cast(NUM_TRAIN_EXAMPLES,dtype=tf.float32))

bnn_model=models.Sequential()

bnn_model.add(layers.Input(shape=(28, 28, 1), name = "input_layer"))

bnn_model.add(tfp.layers.Convolution2DFlipout(32, kernel_size=(3, 3), activation='relu',
name = "hidden_layer_Convflipout_1"))
bnn_model.add(layers.BatchNormalization(name = "normalization"))
bnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name = "pooling"))

bnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name = "max_pooling"))
bnn_model.add(layers.GlobalAveragePooling2D(name = "average_pooling"))

bnn_model.add(layers.Flatten())

bnn_model.add(tfp.layers.DenseFlipout(100, activation = tf.nn.relu, kernel_divergence_fn = kl_divergence_function, name = "hidden_layer_Denseflipout_1"))
bnn_model.add(tfp.layers.DenseFlipout(20, kernel_divergence_fn = kl_divergence_function, activation = tf.nn.relu, name = "hidden_layer_Denseflipout_2"))
bnn_model.add(tfp.layers.DenseFlipout(1,kernel_divergence_fn=kl_divergence_function,activation=tf.nn.softmax,name="output_layer_flipout",))

bnn_model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=["accuracy"])
print(bnn_model.summary())

bnn_model.fit(X_train, y_train[:,0:1], validation_data=(X_test, y_test[:,0:1]), epochs=3)
