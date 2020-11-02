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
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
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
    
outputdir = "/pylon5/as5phnp/tbilling/sandbox/bayesian/"

train_test_file = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/train_test_index_0.npz"

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
    inner = tfp.layers.Convolution2DFlipout(16, kernel_size=(3, 3), activation='relu')(input0)
    """
    tfp.layers.Convolution2DFlipout(filters=16, kernel_size=(3, 3), strides=(1, 1),
            activation='relu', padding='valid',
            data_format='channels_last', dilation_rate=(1, 1),
            activity_regularizer=None,
            kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(), # build Normal distributions with trainable params
            kernel_posterior_tensor_fn=(lambda d: d.sample()), # takes a tfd.Distribution instance and returns a representative value.
            kernel_prior_fn=tfp.layers.default_multivariate_normal_fn, # build multivariate standard Normal distribution
            #kernel_divergence_fn=(lambda q, p, ignore: kl_lib.kl_divergence(q, p)),
            bias_posterior_fn = tfp.layers.default_mean_field_normal_fn(is_singular=True),
            bias_posterior_tensor_fn=(lambda d: d.sample()), bias_prior_fn=None,
            #bias_divergence_fn=(lambda q, p, ignore: kl_lib.kl_divergence(q, p)),
            seed=None
            )(input0)
    """
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    
    inner = tfp.layers.Convolution2DFlipout(32, kernel_size=(3, 3), activation='relu')(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    
    inner = tfp.layers.Convolution2DFlipout(64, kernel_size=(3, 3), activation='relu')(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)
    
    if wedge == False:
        inner = tfp.layers.Convolution2DFlipout(256, kernel_size=(3, 3), activation='relu')(inner)
        inner = BatchNormalization()(inner)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)

        inner = GlobalAveragePooling2D()(inner)
        
        inner = Dropout(0.2)(inner, training=True)
        inner = tfp.layers.DenseFlipout(350, activation='relu')(inner)
    
    else:
        inner = tfp.layers.Convolution2DFlipout(128, kernel_size=(3, 3), activation='relu')(inner)
        inner = BatchNormalization()(inner)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)
        
        inner = tfp.layers.Convolution2DFlipout(128, kernel_size=(3, 3), activation='relu')(inner)
        inner = BatchNormalization()(inner)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)

        inner = GlobalAveragePooling2D()(inner)

        inner = Dropout(0.2)(inner, training=True)
        inner = tfp.layers.DenseFlipout(250, activation='relu')(inner)
    
    inner = Dropout(0.2)(inner, training=True)
    inner = tfp.layers.DenseFlipout(200, activation='relu')(inner)
    
    inner = Dropout(0.2)(inner, training=True)
    inner = tfp.layers.DenseFlipout(100, activation='relu')(inner)
    
    inner = Dropout(0.2)(inner, training=True)
    inner = tfp.layers.DenseFlipout(20, activation='relu')(inner)
    
    output1 = tfp.layers.DenseFlipout(1+1, activation=None)(inner)
    """
    tfp.layers.DenseFlipout(
        units=1 + 1, activation=None, activity_regularizer=None, trainable=True,
        kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
        kernel_posterior_tensor_fn=(lambda d: d.sample()),
        kernel_prior_fn=tfp.layers.default_multivariate_normal_fn,
        #kernel_divergence_fn=(lambda q, p, ignore: kl_lib.kl_divergence(q, p)), bias_pos
        terior_fn=tfp.layers.default_mean_field_normal_fn(is_singular=True),
        bias_posterior_tensor_fn=(lambda d: d.sample()), bias_prior_fn=None,
        #bias_divergence_fn=(lambda q, p, ignore: kl_lib.kl_divergence(q, p)),
        seed=None
    )(inner)
    """
    output = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1],
                         scale=1e-3 + tf.math.softplus(0.01 * t[...,1:])))(output1)
    
    model_flipout = Model(inputs=input0, outputs=output)
    
    # Compile Model
    model_flipout.compile(loss=negloglik,optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.))

    # Summary of model used
    print(model_flipout.summary())
    
    return model_flipout
    

start_time = time.time()
# Start Training
model_flipout = model()

filepath_model = outputdir+"flipout_CNN_model-{epoch:02d}-{loss:.4f}.h5"
filepath_weight = outputdir+"flipout_CNN_weight-{epoch:02d}-{loss:.4f}.h5"
checkpoint_model = ModelCheckpoint(filepath_model, monitor='loss', verbose=1,
                    save_best_only=True, save_weights_only = False, mode='min', save_freq=100)
checkpoint_weight = ModelCheckpoint(filepath_weight, monitor='loss', verbose=1,
                    save_best_only=True, save_weights_only = True, mode='min', save_freq=100)
callbacks_list = [checkpoint_weight, checkpoint_model]

# fit the model
history_flipout = model_flipout.fit(train_images, train_labels, epochs=N_EPOCH, callbacks=callbacks_list,
                                    batch_size=32, validation_split=0.1, verbose = 2, shuffle=True)

running_time = time.time() - start_time
print("Finish Training CNN in ", str(timedelta(seconds=running_time)))
        
# Save model information
if wedge == True:
    # Save model
    print("saving model trained on wedge filtered data ...")
    model_flipout.save_weights(outputdir+"flipout_CNN_weights_wedge.h5")
    model_flipout.save(outputdir+"flipout_CNN_model_wedge.h5")
    
    # Save history
    print("Removing Scaling factor ({}) and saving histories...".format(factor))
    history_keys = np.array(list(history_flipout.history.keys()))
    np.savez(outputdir+"flipout_CNN_wedge_history",loss=np.array(history_flipout.history[str("loss")])/factor, val_loss=np.array(history_flipout.history[str("val_loss")])/factor)
                 
if wedge == False:
    # Save model
    print("saving model trained on nowedge filtered data ...")
    model_flipout.save_weights(outputdir+"flipout_CNN_weights_nowedge.h5")
    model_flipout.save(outputdir+"flipout_CNN_model_nowedge.h5")

    # Save history
    print("Removing Scaling factor ({}) and saving histories...".format(factor))
    history_keys = np.array(list(history_flipout.history.keys()))
    np.savez(outputdir+"flipout_CNN_nowedge_history",loss=np.array(history_flipout.history[str("loss")])/factor, val_loss=np.array(history_flipout.history[str("val_loss")])/factor)

# evaluate trained model
test_loss = model_flipout.evaluate(test_images, test_labels)

# make predictions
flipout_predictions_list = [[model_flipout(test_images) for _ in range(500)]] # (500, 100, 1) = (# of masks, # of datasets, # of classes)
flipout_predictions = np.concatenate(flipout_predictions_list, axis=1) # (100, 500)

# select an index from the 200 prediciton over 500 dropout masks
idx = 50
p0 = np.array([p[idx] for p in flipout_predictions])
print("posterior mean: {}".format(p0.mean(axis=0)))
print("true label: {}".format(test_labels[idx]/factor))
print()

# probability and variance
for i, (prob, var) in enumerate(zip(p0.mean(axis=0), p0.std(axis=0))):
    print("value: {}; probability: {:.1%}; var: {:.2%} ".format(i, prob, var))
    
# ???? Plot a 2D histogram ???? https://matplotlib.org/3.1.1/gallery/statistics/hist.html
fig, ax = plt.subplots(tight_layout=True)
hist = ax.hist2d(x, y, bins=10)

# look at the Probability distributions of the monte carlo predictions and in blue you see the prediction of the ensemble
plt.figure(figsize=(12,12))
plt.hist(p0[:,i], bins=100, density=True)
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
if wedge == True:
    plt.savefig(outputdir+"MC_PDF_wedge.png")
if wedge == False:
plt.savefig(outputdir+"MC_PDF_nowedge.png")

# one-to-one with errorbars (Fig 11) https://arxiv.org/pdf/1911.08508.pdf
results = glob.glob(outputdir+"*.npy")
result=np.load(results[0])

# Convert to true tau units
true_tau = low_z_tau + h_2 * result["truth"][:,n]/factor
predicted_tau = low_z_tau + h_2 * result["prediction"][:,n]/factor

plt.figure(figsize=(12,12))
plt.errorbar(true_tau, predicted_tau, xerr=xerr, yerr=yerr, fmt='-o')
#plt.scatter(true_tau, predicted_tau, s=6, lw=0, alpha=0.9, label=fold[r])
x = np.linspace(0.95*np.min(true_tau), 1.05*np.max(true_tau), 1000)
plt.plot(x, x, 'k--',lw=1,alpha=0.2)
plt.xlabel()
plt.ylabel()
