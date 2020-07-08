# -*- coding: utf-8 -*-
# needs to be ran in dp-GPU environment with Tensorflow version
# tf.__version__ : '1.8.0'

from __future__ import print_function, division, absolute_import

from memory_profiler import profile

"""
For some reason you cannot run the script with the memory_profile while you are
in the cwd. You have to change directories.  run:

python -m memory_profiler memory_profiler/10_fold_validation2_memory_profiler.py

OR

mprof run memory_profiler/10_fold_validation2_memory_profiler.py mprof plot
"""

# Imports

import json, os, sys, time, math, matplotlib, h5py, glob, gc
import numpy as np
import tensorflow as tf

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

from matplotlib import cm
from matplotlib import gridspec
from sklearn.model_selection import KFold
from scipy.ndimage import gaussian_filter


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import backend as K

# from keras.wrappers.scikit_learn import train_test_split

import gc
gc.enable()

tf.logging.set_verbosity(tf.logging.INFO)

data_path = "/pylon5/as5phnp/tbilling/data/"
reionfilename = data_path + "t21_snapshots_nowedge_v8.hdf5"
inputFile = reionfilename
outputdir = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/memory_profiler/"

steps = 4
factor = 1000.0
nfold = 10

# Memory profiles
precision = 10

# write out
fp = open(outputdir + "memory_profiler_cnn.log", "w+")


@profile(precision=precision, stream=fp)
def readLabels(ind=None, **params):
    """
    Read in snapshot "labels" (i.e., parameters to regress).

    This function is meant to be used with batches.

    Parameters
    ----------
    ind : tuple of ints, optional
        The indices corresponding to the parameters to read in. If None, then
        all parameters are read in.
    params : dict
        A dictionary defining other important parameters.

    Returns
    -------
    labels : ndarray
        An array of size (N_realizations, N_parameters) containing the
        values to be regressed on.
    """
    with h5py.File(inputFile, "r") as h5f:
        if ind is None:
            # read in all parameters, size (N_realizations, N_parameters)
            labels = np.asarray(
                f["Data"][u"snapshot_labels"]
            )
        else:
            # read in just the specified ones
            labels = np.asarray(f["Data"][u"snapshot_labels"][:, ind])

    if labels.ndim == 1:
        print("training on just one param.")
        print("starting with the following shape, dim:", labels.shape, labels.ndim)
        if labels.ndim > 1:
            labels = labels[:, params["predictoneparam"]]
    elif labels.shape[1] == 2:
        print("training on two params.")
        print("starting with the following shape, dim:", labels.shape, labels.ndim)
        if labels.ndim > 1:
            labels = labels[:, ind]

    # if there's only one label per image, we'll have to reshape it:
    if labels.ndim == 1:
        print("reshaping data...")
        labels = labels.reshape(-1, 1)

    return labels


@profile(precision=precision, stream=fp)
def readImages(ind, **params):
    """
    Read in the snapshot images.

    This function will read in the data that will be used as input data. It will
    also transpose the data, which are on disk as (N_snapshot, N_redshift,
    N_pix, N_pix), to (N_snapshot, N_pix, N_pix, N_redshift). In other words,
    the redshift axis will be reordered to be the fastest-running index, so that
    it can easily be treated as the "color channel" index for the CNN.

    Parameters
    ----------
    ind : tuple of ints
        The indices to extract from on disk. These should be passed in such a
        way that they are conducive to indexing a numpy array.
    params : dict
        A dictionary defining other important parameters.

    Returns
    -------
    data : ndarray
        An array containing the input data. It has shape (N_snapshot, N_pix,
        N_pix, N_redshift).
    tuple of ints
        The shape of an individual snapshot.
    """

    print("reading data from", inputFile)

    with h5py.File(inputFile, "r") as h5f:
        if "crop" in params:
            # use just the top corner of the images
            data = np.asarray(
                f["Data"][u"t21_snapshots"][ind, :, 0 : params["crop"], 0 : params["crop"]]
            )
        else:
            # use everything!
            print("reading all data", len(ind))
            # resulting shape: (N_realizations, N_redshifts, N_pix, N_pix)
            data = np.asarray(
                f["Data"][u"t21_snapshots"][ind, :, :, :]
            )

    print("finished loading data.", data.shape)

    # transpose to (N_realizations, N_pix, N_pix, N_redshifts)
    data = data.transpose(0, 2, 3, 1)

    return data, data[0].shape


@profile(precision=precision, stream=fp)
def Mean_Squared_over_true_Error(y_true, y_pred):
    """
    Define the relative mean squared error loss function for Keras.

    This function defines a custom loss function for Keras of the mean squared
    error divided by the true value. This helps regularize the loss function
    in cases when the true values are much larger or smaller than 1.

    Parameters
    ----------
    y_true : Keras tensor
        The "true" values from Keras.
    y_pred : float or Keras tensor
        The "predicted" values from Keras.

    Returns
    -------
    loss : Keras tensor
        The resulting loss function that is suitable to be used in a Keras
        model.
    """
    if not K.is_keras_tensor(y_pred):
        y_pred = K.constant(y_pred)

    y_true = K.cast(y_true, y_pred.dtype)
    diff_ratio = K.square((y_pred - y_true) / K.clip(K.abs(y_true), K.epsilon(), None))
    loss = K.mean(diff_ratio, axis=-1)
    # Return a function

    return loss


@profile(precision=precision, stream=fp)
def r2_keras(y_true, y_pred):
    """
    Compute the R2 metric in Keras.

    Parameters
    ----------
    y_true : Keras tensor
        The "true" values from Keras.
    y_pred : Keras tensor
        The "predicted" values from Keras.

    Returns
    -------
    Keras tensor
        The R2 value which can be used as a Keras metric.
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


@profile(precision=precision, stream=fp)
def makeModel(input_shape, Nregressparams):
    """
    Make the Keras model for a CNN application.

    Parameters
    ----------
    input_shape : tuple of ints
        The shape of a single snapshot of the input data.
    Nregressparams : int
        The number of output parameters to regress on.

    Returns
    -------
    model : Keras model
        The resulting compiled Keras model to be trained.
    """
    model = Sequential()
    model.add(
        Conv2D(
            16,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            input_shape=input_shape,
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.2))
    model.add(Dense(200, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(1))

    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.001, decay=0.0),
        loss=Mean_Squared_over_true_Error,
    )
    print(model.summary())
    return model


@profile(precision=precision, stream=fp)
def savePreds(model, eval_data, eval_labels, Ntot, fold, istart=0, outdir=None):
    """
    Save a series of predictions from a trained Keras model.

    Parameters
    ----------
    model : Keras model
        The trained Keras model to use for prediction.
    eval_data : ndarray
        The input data to use for the predictions.
    eval_labels : ndarray
        The actual labels for the prediction data for comparison.
    Ntot : int
        The total number of snapshots to predict on.
    fold : int
        The fold of training data corresponding to the model.
    istart : int, optional
        The starting index for the input data.
    outdir : str, optional
        The output directory to write a file to. If not specified, defaults to
        the current directory.

    Returns
    -------
    None

    Notes
    -----
    This function will save a numpy .npy file with the name "results<n>.npy",
    where `n` corresponds to the fold number. This file contains three fields,
    "truth", "prediction", and "fold", containing the true value, the predicted
    values, and the fold they correspond to, respectively.
    """
    outputFile = os.path.join(outdir, "results{:d}.npy".format(fold))

    # The Predict() method - is for the actual prediction. It generates output
    # predictions for the input samples.
    preds = model.predict(eval_data, verbose=0).flatten()  # 0 = silent

    Nregressparams = len(eval_labels[0])

    results = np.zeros(
        (Ntot, Nregressparams),
        dtype=[("truth", "f"), ("prediction", "f"), ("fold", "i")],
    )
    iend = istart + len(eval_labels)

    results["fold"][istart:iend] = fold
    results["truth"][istart:iend] = eval_labels
    for n in range(Nregressparams):
        results["prediction"][istart:iend, n] = preds[n::Nregressparams]

    np.save(outputFile, results)


@profile(precision=precision, stream=fp)
def cross_validation():
    """
    Run a cross-validation and N-fold training of a ML network.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    params = {
        "runlabel": "alldata_b_",
        "Nfolds": nfold,
        "debug": False,
        "epochs": steps,
        "crop": 512,
        "predicttwoparams": [1],  # [0, 1],
        "patience": 20,
        "learning_rate": 0.001,
        "decay": True}

    kfold_split = sorted(glob.glob("/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/train_test_index_*.npz"))
    scores = []
    istart = 0

    for i in np.arange(nfold):
        # Load Index Label
        train_index = np.load(kfold_split[i])["train_index"]
        test_index = np.load(kfold_split[i])["test_index"]
        # Load Labels and Images
        trainlabels = readLabels(ind=None)[train_index, 5] * factor
        trainlabels = trainlabels.reshape(-1, 1)
        images, shape_label = readImages(ind=train_index)

        testlabels = readLabels(ind=None)[test_index, 5] * factor
        testlabels = testlabels.reshape(-1, 1)
        test_image, input_shape = readImages(ind=test_index)

        fold = i
        log_dir = os.path.join(outputdir, "output", str(fold))
        cb = keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=10, write_images=True
        )

        print("making model...")
        print("compiling model...")
        print("number of regression parameters: ", trainlabels.shape[1])
        model = makeModel(input_shape, Nregressparams=trainlabels.shape[1])

        # model = load_model(outputdir+'hyperParam_model.h5')
        # model = load_model(outputdir+'hyperParam_model.h5',custom_objects={"r2_keras":r2_keras})

        # The fit() method - Trains the model with the given inputs (and
        # corresponding training labels)
        print("fitting model...")
        print("-" * 150)
        print("*" * 150)
        print("-" * 150)
        # print start time
        os.system("date")
        history = model.fit(
            images,
            trainlabels,
            batch_size=32,
            verbose=2,
            validation_split=0.2,
            epochs=steps,
            callbacks=[cb],
        )
        os.system("date")
        print("-" * 150)
        print("*" * 150)
        print("-" * 150)

        # save model
        print("saving model...")
        model.save(outputdir + "model_fold{}.h5".format(fold))
        model.save_weights(outputdir + "model_weights_fold{}.h5".format(fold))

        # The evaluate() method - gets the loss statistics on already trained
        # model using the validation (or test) data and the corresponding
        # labels. Returns the loss value and metrics values for the model.
        print("calculating test loss...")
        score = model.evaluate(
            test_image, testlabels, batch_size=32, verbose=1
        )  # 1 = progress bar
        # returns: loss
        print("          Test loss:", score)
        print("")
        # loss, mse, mae, mape
        scores.append(score)

        # Save history
        print("Removing Scaling factor ({}) and saving histories...".format(factor))
        history_keys = np.array(list(history.history.keys()))
        for key in history_keys:
            np.savez(
                outputdir + "history_{}_{:d}".format(str(key), fold),
                metric=np.array(history.history[str(key)]) / factor,
            )

        # save predictions
        print("saving and plotting predictions for fold", fold + 1, "...")
        if fold == 0:
            results = None
        savePreds(
            model,
            test_image,
            testlabels,
            len(trainlabels),
            fold,
            istart=istart,
            outdir=outputdir,
        )
        istart += len(test_index)

        # destroy the current TF graph and creates a new one
        print("removing model history and TF graph...")
        del images
        del test_image
        del model
        K.clear_session()
        gc.collect()

        # is this supposed to be closed inside the loop? or after all the loops
        # are finished?
        fp.close()

    return


if __name__ == "__main__":

    cross_validation()
