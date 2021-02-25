# needs to be ran in hp_opt environment with Tensorflow version
#  https://stackoverflow.com/questions/58566096/custom-loss-function-that-updates-at-each-step-via-gradient-descent

# tf.__version__ : '2.1.0'

from __future__ import print_function, division, absolute_import

import numpy as np, matplotlib.pyplot as plt, gc, time, h5py, keras, random, os
import tensorflow as tf

from keras import backend as K
from datetime import timedelta
from tensorflow.keras import Model
from sklearn.metrics import accuracy_score
from matplotlib.ticker import PercentFormatter

gc.enable()

# As you are trying to use function decorator in TF 2.0, please enable run function eagerly by using below line after importing
# TensorFlow: https://www.tensorflow.org/guide/effective_tf2#use_tfconfigexperimental_run_functions_eagerly_when_debugging
tf.config.experimental_run_functions_eagerly(True)

data_path = "/lustre/aoc/projects/hera/tbilling/ml/data/" #"/pylon5/as5phnp/tbilling/data/"

train_test_file = data_path+"train_test_index_80_20_split.npz"

n=0
factor =1000.

# Loss Function
negloglik = lambda y_true, y_pred: -y_pred.log_prob(y_true)
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


for wedge in [False, True]:

    if wedge == False:
        print("No wedge Filtered Data...")
        inputFile = data_path+"t21_snapshots_nowedge_v12.hdf5"
        outputdir = "/lustre/aoc/projects/hera/tbilling/ml/likelihood/" #"/pylon5/as5phnp/tbilling/sandbox/bayesian/dropout/nowedge/"
        model = outputdir + "dopout_CNN_model_nowedge.h5"

    if wedge == True:
        print("Wedge Filtered Data...")
        inputFile = data_path+"t21_snapshots_wedge_v12.hdf5"
        outputdir = "/lustre/aoc/projects/hera/tbilling/ml/likelihood/" #"/pylon5/as5phnp/tbilling/sandbox/bayesian/dropout/wedge/"
        model = outputdir + "dopout_CNN_model_wedge.h5"

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

        if 'crop' in params:
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

    # Load Index Label
    test_index = np.load(train_test_file)["test_index"]

    # Load images and labels for testing
    test_labels = readLabels(ind=None)[test_index,5]*factor
    test_labels = test_labels.reshape(-1, 1)
    test_images,input_shape = readImages(ind=test_index)

    # Load Model
    model_dropout = keras.models.load_model(model, custom_objects={"Mean_Squared_over_true_Error": Mean_Squared_over_true_Error})
    print(model_dropout.summary())

    # evaluate trained model
    test_loss = model_dropout.evaluate(test_images, test_labels)

    # make predictions
    dropout_predictions = []
    for i in range(500):
        #y_p = model_dropout.predict(test_images, batch_size=test_labels.shape[0])
        y_p = model_dropout.predict(test_images, batch_size=test_labels.shape[0])
        dropout_predictions.append(y_p) # (500, 200, 1) = (# of masks, # of datasets, # of classes)
        # np.array(dropout_predictions)[:,0,0]
    np.savez(outputdir+"predictions", prediciton=np.array(dropout_predictions))


"""
pred = np.array(dropout_predictions).flatten()
Nregressparams = len(test_labels)
Ntot = len(test_labels)

results = np.zeros((Ntot, Nregressparams),
                       dtype = [('truth', 'f'), ('prediction', 'f'), ('fold', 'i')])
iend = istart+len(test_labels)

#print('istart and iend', istart, iend)

results['fold'][istart:iend] = fold
results['truth'][istart:iend] = test_labels

for n in range(Nregressparams):
    results['prediction'][istart:iend,n] = preds[n::Nregressparams]
    #results['prediction'][istart-100:iend-100,n] = preds[n::Nregressparams]

np.save(outputdir+"eval_pred_results.npy", results)
"""

"""
# select an index from the 200 prediciton over 500 dropout masks
idx = 50
p0 = np.array([p[idx] for p in dropout_predictions])
print("posterior mean: {}".format(p0.mean(axis=0)))
print("true label: {}".format(test_labels[idx]/factor))
print()

# probability and variance
for i, (prob, var) in enumerate(zip(p0.mean(axis=0), p0.std(axis=0))):
    print("class: {}; probability: {:.1%}; var: {:.2%} ".format(i, prob, var))
    
# Plot a 2D histogram ???? https://matplotlib.org/3.1.1/gallery/statistics/hist.html
fig, ax = plt.subplots(tight_layout=True)
hist = ax.hist2d(x, y, bins=10)

# look at the Probability distributions of the monte carlo predictions and in blue you see the prediction of the ensemble
plt.figure(figsize=(12,12))
plt.hist(p0[:,i], bins=100, density=True)
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
ax.set_title(f"class {i}")
ax.label_outer()

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
"""
