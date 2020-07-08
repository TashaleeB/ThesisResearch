from __future__ import print_function, division, absolute_import

# Imports
import json, os, sys, time, math, h5py
import numpy as np

#random seed to control the reproducability
seed = 8675309
np.random.seed(seed)
#np.random.seed(datetime.datetime.now().microsecond)

import tensorflow as tf
import pandas as pd
import keras

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

from matplotlib import cm
from matplotlib import gridspec
from sklearn.model_selection import KFold
from scipy.ndimage import gaussian_filter

from keras.datasets import mnist
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import backend as K
#from keras.wrappers.scikit_learn import train_test_split


tf.logging.set_verbosity(tf.logging.INFO)

data_path = '/pylon5/as5phnp/tbilling/data/'
reionfilename = data_path+'t21_snapshots_nowedge_v7.hdf5'
inputdir = '/pylon5/as5phnp/tbilling/400/newlossfun/tester/'
inputFile = reionfilename
outputdir = inputdir

steps = 1
factor =1000.
nfold = 4


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
        data = np.asarray(f['Data'][u't21_snapshots'][ind,:,0:params['crop'],0:params['crop']])
    else:
        #use everything!
        print('reading all data', len(ind))
        data = np.asarray(f['Data'][u't21_snapshots'][ind,:,:,:]) # (N_realizations, N_redshifts, N_pix, N_pix)
        #print('loaded data', len(ind))

    print('finished loading data.', data.shape)

    data  = data.transpose(0,2,3,1) #(N_realizations, N_pix, N_pix, N_redshifts)

    return data, data[0].shape



def plotOneDataSet(data, label, snapshotnum, title, vmin, vmax, width = 7,
                   addzlabel = False, **params):
    gs = gridspec.GridSpec(1,width)

    zList = None

    for j in range(width):
        n = j+4 #I want to plot snapshots 6, 7, 8,...
        ax = plt.subplot(gs[0, j])
        cmap = plt.get_cmap('plasma')
        ax.imshow(data[:,:,n], cmap=makeCustomColormap(**params),
                      origin='lower', interpolation='none',
                      vmin = vmin, vmax=vmax)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_xticklines(), visible=False)
        plt.setp(ax.get_yticklines(), visible=False)
        if addzlabel == True:
            ax.set_xlabel(r'$z='+'{0:.1f}'.format(zList[n])+'$', fontsize=10)

    #plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(outputdir + '/output/' + params['runlabel'] + '/' + str(snapshotnum) +
                '_label='+str(label[0])+'.png',
                transparent = True, bbox_inches='tight')
    plt.clf()
    return None

# custom R2-score metrics for keras backend
from keras import backend as K

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def makeModel(input_shape, Nregressparams, flatten=False):
    """Model function for the CNN.  Need more layers?  This is where you add them"""


    #this is a regression model that, at first at least, was 
    #based on https://navoshta.com/end-to-end-deep-learning/
    #which was based on an arxiv paper 1604.07316
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if flatten:
        model.add(Flatten())
    else:
        model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(Nregressparams))
    print(model.summary())
    return model

def plot_model(model):
    plot_model(model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)
    return

def Mean_Squared_over_true_Error(y_true, y_pred):
    # Create a custom loss function that divides the difference by the true
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)

    y_true = K.cast(y_true, y_pred.dtype)
    diff_ratio = K.square((y_pred - y_true)/K.clip(K.abs(y_true),K.epsilon(),None))
    loss = K.mean(diff_ratio, axis=-1)
    # Return a function
    return loss

def compileModel(model, lr=0.01, decay=0.):
    model.compile(loss= Mean_Squared_over_true_Error,
                  optimizer=keras.optimizers.Adam(lr=lr, decay=decay)
                  metrics=[r2_keras])
    return None

def loadBatchData(b, train_index, **params):
    """
    reads in one batch and returns the data
    """
    #this will start with the bth index and choose every Nth, where N = params['batches']
    ind = train_index[b::params['batches']]
    return ind, readImages(ind, **params)[0]


def savePreds(model, eval_data, eval_labels, Ntot, fold, istart=0, outdir=None):
    outputFile = os.path.join(outputdir, "results{}_{:d}.npy".format(steps,fold))

    # The Predict() method -  is for the actual prediction. It generates output predictions for the input samples.
    preds = model.predict(eval_data, verbose=1).flatten()

    Nregressparams = len(eval_labels[0])

    results = np.zeros((Ntot, Nregressparams),
                           dtype = [('truth', 'f'), ('prediction', 'f'), ('fold', 'i')])
    iend = istart+len(eval_labels)

    #print('istart and iend', istart, iend)

    results['fold'][istart:iend] = fold
    results['truth'][istart:iend] = eval_labels
    for n in range(Nregressparams):
        results['prediction'][istart:iend,n] = preds[n::Nregressparams]

    np.save(outputFile, results)

def plotPreds(outdir=None):
    results_file = os.path.join(outputdir, "results{}_{:d}.npy".format(steps,fold))
    results = np.load(results_file)
    Nregressparams = len(results['prediction'][0])

    for n in range(Nregressparams):
        plt.scatter(results['truth'][:,n], results['prediction'][:,n], s=2, lw=0, alpha=0.5, c='k')
        #add a 1-to-1 line:
        x = np.linspace(0.95*np.min(results['truth'][:,n]),
                        1.05*np.max(results['truth'][:,n]), 100)
        plt.plot(x, x, c='r', ls='--')
        plt.xlabel('true')
        plt.ylabel('predicted')
        filename = "results{}.npy".format(steps)
        if outdir is None:
            outfile = os.path.join(outputdir, 'output', filename)
        else:
            outfile = os.path.join(outdir, filename)
        plt.savefig(outfile)
        plt.clf()
    for n in range(Nregressparams):
        plt.scatter(results['truth'][:,n], results['prediction'][:,n], s=2, lw=0, alpha=0.5, c='k')
        plt.xlabel('true')
        plt.ylabel('predicted')
        filename = 'PredvsTruth_{}.png'.format(steps)
        plt.savefig(outfile)
        plt.clf()

    return

def crossvalidate(labels, input_shape, data=None, valfrac=0.2, outdir=None, **params):
    #loop through the folds, training on 80% of Nfolds-1, testing on 20% of Nfolds-1,
    #and predicting the remaining fold.
    #valfrac sets the 80/20 split

    params = {'runlabel':'alldata_b_',
            'Nfolds': nfold,
            'debug': False,
            'epochs': steps,
            'crop': 512,
            'predicttwoparams': [1],# [0, 1],
            'patience': 20,
            'learning_rate': 0.1,
            'decay': True}
    scores = []
    kf = KFold(n_splits=params['Nfolds'], random_state=seed, shuffle=True)
    istart = 0
    for fold, [train_index, test_index] in enumerate(kf.split(np.arange(len(labels)))):
        print('regressing fold', fold+1, 'of', params['Nfolds'], '...')

        print("Train Index: ", train_index, "\n")
        print("Test Index: ", test_index)
        
        log_dir = os.path.join(outputdir, 'output', str(fold))
        cb = keras.callbacks.TensorBoard(log_dir=log_dir,
                                         histogram_freq=10, write_images=True)

        #make and compile a new keras model for this fold:
        print('making model...')
        print('number of regression parameters: ', labels.shape[1])
        model = makeModel(input_shape, labels.shape[1], flatten=False)
        print('compiling model...')
        if params['decay']:
            epochs = params['epochs']
            lr = params['learning_rate']
            # decay down to nearly 0 by the end of training
            decay = lr / epochs
            compileModel(model, lr, decay)
        else:
            lr = params['learning_rate']
            compileModel(model, lr, decay=0.)

        # The fit() method - Trains the model with the given inputs (and corresponding training labels)
        print('fitting model...')
        # Stored your model.fit results in a 'history' variable
        history = model.fit(data[train_index], labels[train_index], batch_size=32, verbose=2,
                            validation_split=0.2, epochs=steps, callbacks=[cb])
        
        """
        # Plot training & validation accuracy values
        plt.figure(figsize=(15,10))
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        """
        # Plot training & validation loss values
        plt.figure(figsize=(15,10))
        plt.plot(history.history['loss'][1:],".-")
        plt.plot(history.history['val_loss'][1:],".-")
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.savefig(outputdir+'Loss_Epoch_linearity{}_{:d}.png'.format(steps,fold))
        plt.clf()
        
        testData = data[test_index]

        # The evaluate() method - gets the loss statistics on already trained model using the validation (or test) data and the corresponding labels. Returns the loss value and metrics values for the model.
        print('calculating test loss...')
        score = model.evaluate(testData, labels[test_index], batch_size=32, verbose=1)
        # returns: loss
        print('          Test loss:', score)
        print('')
        scores.append(score)
        
        #save predictions
        #print('saving and plotting predictions for fold', fold+1, '...')
        if fold == 0:
            results = None
        savePreds(model, testData, labels[test_index], len(labels), fold, istart=istart, outdir=outdir)
        istart += len(test_index)

        # save model
        model_out = os.path.join(outdir, 'fold_{:d}.model'.format(fold))
        print('saving model {}...'.format(model_out))
        model.save(model_out)
        
        n = 0
        results = np.load(outputdir+"results{}_{:d}.npy".format(steps,fold))
        plt.figure(figsize=(10,8))
        plt.scatter(results['truth'][:,n]/factor, results['prediction'][:,n]/factor, s=2, lw=0, alpha=0.5, c='k')
        plt.xlabel('true')
        plt.ylabel('predicted')
        filename = outputdir+'PredvsTruth{}_{:d}.png'.format(steps,fold)
        plt.savefig(filename)
        plt.clf()
        
        # Save history
        print("Removing Scaling factor ({}) and saving history...".format(factor))
        np.savez(outputdir+"history{}_{:d}".format(steps,fold),
                 val_loss=np.array(history.history['val_loss'])/factor,
                 loss=np.array(history.history['loss'])/factor)

        # Plot truth and prediction with target
        plt.figure(figsize=(15,10))
        plt.scatter(results['truth'][:,n]/factor, results['prediction'][:,n]/factor, s=2, lw=0, alpha=0.5, c='k')
        #add a 1-to-1 line: 
        x = np.linspace(0.95*np.min(results['truth'][:,n]/factor),
                        1.05*np.max(results['truth'][:,n]/factor), 100)
        plt.plot(x, x, "r--")#c='r', ls='r--')
        plt.xlabel('true') 
        plt.ylabel('predicted')
        #plt.legend(markerscale=5,loc=0, fontsize=14)
        filename = outputdir+"results{}_with_target.png".format(fold)
        plt.savefig(filename) 
        plt.clf() 

        # test resetting model
        K.clear_session()

labels = readLabels(ind=None)[:,5]*factor
labels = labels.reshape(-1, 1)
images,shape =readImages(ind=np.arange(1000))
labels=labels; input_shape=shape; data=images; valfrac=0.2; outdir=outputdir
crossvalidate(labels=labels, input_shape=shape, data=images, valfrac=0.2, outdir=outputdir)


In [51]: fv6 = h5py.File(v6, 'r')
    ...: labelsv6 = np.asarray(fv6['Data'][u'snapshot_labels'][:,5])
    ...:
    ...: fv7 = h5py.File(v7, 'r')
    ...: labelsv7 = np.asarray(fv7['Data'][u'snapshot_labels'][:,5])
    ...:
    ...: plt.figure(figsize=(15,10))
    ...: # Normalize by setting density to true
    ...: kwargs1 = dict(alpha=0.75, bins=100, density=None, stacked=True, histtype='step')
    ...: kwargs2 = dict(alpha=0.25, bins=100, density=None, stacked=True, histtype='stepfilled')
    ...:
    ...: # Plot
    ...: plt.hist(labelsv6, **kwargs1, color='r', label='v6 Tau Label')
    ...: plt.hist(labelsv7, **kwargs2, color='b', label='v7 Tau Label')
    ...:
    ...: plt.gca().set(title='', xlabel='Tau', ylabel='Count')
    ...: plt.xlim(0,0.18)
    ...: plt.ylim(0,22)
    ...: plt.legend()
    ...: plt.savefig("hist_v6_and_v7.png")



# Split data up
kf = KFold(n_splits=5, random_state=seed, shuffle=True)
KF =np.array(list(kf.split(np.arange(len(labels)))))
[train_index, test_index] = KF[0]

train_label = labels[train_index]
train_images = images[train_index]

test_label = labels[test_index]
test_images = images[test_index]
#labels=labels; input_shape=shape; data=images; valfrac=0.2; outdir=outputdir


