#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Usage: regress 

from __future__ import print_function, division, absolute_import

# Imports
import json, os, sys, time, math, matplotlib, h5py
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
reionfilename = data_path+'t21_snapshots_nowedge.hdf5'
inputdir = '/pylon5/as5phnp/tbilling/wtf/'
inputFile = reionfilename
outputdir = inputdir

steps = 4
factor =1000.

def ensureDirectory(file_path):
    """
    figure out if a directory exists, and if it doesn't exist yet, make it!
    """
    directory = os.path.abspath(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def makeCustomColormap(**params):
    c = mcolors.ColorConverter().to_rgb

    #monochromatic green
    #customcmap = make_colormap([c('#c7d1aa'), c(colorGreen), 0.5, c(colorGreen), c('#455519')])
    #diverging red-to-blue:
    if theme == 'poster':
        customcmap = make_colormap([c('0.95'), c(colorBlue), c(colorDkTeak)])
    elif theme == 'paper':
        customcmap = make_colormap([c('w'), c(colorRed)])
    else:
        customcmap = make_colormap([c(colorBlue), c('w'), 0.5, c('w'), c(colorRed)])
    return customcmap

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
    model.add(Dense(Nregressparams)) #maybe add softmax activation layer
    print(model.summary())
    return model

def plot_model(model):
    plot_model(model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)
    return

def compileModel(model, lr=0.01, decay=0.):
    model.compile(loss='mean_squared_error',
                  optimizer=keras.optimizers.Adam(lr=lr, decay=decay))
    return None

def loadBatchData(b, train_index, **params):
    """
    reads in one batch and returns the data
    """
    #this will start with the bth index and choose every Nth, where N = params['batches']
    ind = train_index[b::params['batches']]
    return ind, readImages(ind, **params)[0]


def savePreds(model, eval_data, eval_labels, Ntot, fold, istart=0, outdir=None):

    outputFile = os.path.join(outputdir+"results{}_{}.npy".format(params['epochs'],fold))

    preds = model.predict(eval_data, verbose=0).flatten()

    Nregressparams = len(eval_labels[0])

    if fold == 0:
        results = np.zeros((Ntot, Nregressparams),
                           dtype = [('truth', 'f'), ('prediction', 'f'), ('fold', 'i')])
    else:
        results = np.load(outputFile)
    iend = istart+len(eval_labels)

    #print('istart and iend', istart, iend)

    results['fold'][istart:iend] = fold
    results['truth'][istart:iend] = eval_labels
    for n in range(Nregressparams):
        results['prediction'][istart:iend,n] = preds[n::Nregressparams]

    np.save(outputFile, results)

def plotPreds(outdir=None):
    if outdir is None:
        results_file = os.path.join(outputdir, 'output', params['runlabel'], "results{}.npy".format(params['epochs']))
    else:
        results_file = os.path.join(outdir, "results{}.npy".format(params['epochs']))
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
        filename = "results{}.npy".format(params['epochs'])
        if outdir is None:
            outfile = os.path.join(outputdir, 'output', params['runlabel'], filename)
        else:
            outfile = os.path.join(outdir, filename)
        plt.savefig(outfile)
        plt.clf()
    for n in range(Nregressparams):
        plt.scatter(results['truth'][:,n], results['prediction'][:,n], s=2, lw=0, alpha=0.5, c='k')
        plt.xlabel('true')
        plt.ylabel('predicted')
        filename = 'PredvsTruth_{}.png'.format(params['epochs'])
        plt.savefig(outfile)
        plt.clf()

    return

def crossvalidate(labels, input_shape, data=None, valfrac=0.2, outdir=None, **params):
    #loop through the folds, training on 80% of Nfolds-1, testing on 20% of Nfolds-1,
    #and predicting the remaining fold.
    #valfrac sets the 80/20 split

    params = {'runlabel':'alldata_b_',
            'Nfolds': 2,
            'debug': False,
            'epochs': steps,
            'crop': 512,
            'predicttwoparams': [1],# [0, 1],
            'patience': 20,
            'learning_rate': 0.1,
            'decay': True}
    
    kf = KFold(n_splits=params['Nfolds'], random_state=seed, shuffle=True)
    istart = 0
    for fold, [train_index, test_index] in enumerate(kf.split(np.arange(len(labels)))):
        print('regressing fold', fold+1, 'of', params['Nfolds'], '...')
        if outdir is None:
            log_dir = os.path.join(outputdir, 'output', params['runlabel'], str(fold))
        else:
            log_dir = os.path.join(outdir, str(fold))
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

        #fit the model
        print('fitting model...')
        # Stored your model.fit results in a 'history' variable
        history = model.fit(data[train_index], labels[train_index], batch_size=32, verbose=2,
                            validation_split=0.2, epochs=params['epochs'], callbacks=[cb])
        #savehistory(hist_dict=history)
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
        plt.savefig(outputdir+'Loss_Epoch_linearity{}.png'.format(params['epochs']))
        testData = data[test_index]
        plt.clf()

        #print out a summary stat:
        print('calculating test loss...')
        score = model.evaluate(testData, labels[test_index], batch_size=32, verbose=1)

        print('          Test loss:', score)
        print('')

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
        plt.figure(figsize=(15,10))
        results = np.load(outputdir+"results{}_{}.npy".format(params['epochs'],fold))
        plt.scatter(results['truth'][:,n]/factor, results['prediction'][:,n]/factor, s=4, lw=0, alpha=0.5, c='k')
        plt.xlabel('true')
        plt.ylabel('predicted')
        filename = outputdir+'PredvsTruth{}.png'.format(params['epochs'])
        plt.savefig(filename)
        plt.clf()
        
        plt.figure(figsize=(15,10))
        x = np.linspace(0.95*np.min(results['truth'][:,n]/factor),1.05*np.max(results['truth'][:,n]/factor), 100)
        plt.scatter(results['truth'][:,n]/factor, results['prediction'][:,n]/factor, s=4, lw=0, alpha=0.5, c='k')
        plt.plot(x, x, "r--")#c='r', ls='r--')
        plt.xlabel('true')
        plt.ylabel('predicted')
        filename = outputdir+'results{}_with_target.png'.format(params['epochs'])
        plt.savefig(filename)
        plt.clf()

        # Save history
        print("Removing Scaling factor ({}) and saving history...".format(factor))
        np.savez(outputdir+"history{}steps".format(params['epochs']),
                 val_loss=np.array(history.history['val_loss'])/factor,
                 loss=np.array(history.history['loss'])/factor)

        # test resetting model
        K.clear_session()

labels = readLabels(ind=None)[:,5]*factor
labels = labels.reshape(-1, 1)
images,shape =readImages(ind=np.arange(1000))
labels=labels; input_shape=shape; data=images; valfrac=0.2; outdir=outputdir
crossvalidate(labels=labels, input_shape=shape, data=images, valfrac=0.2, outdir=outputdir)

def crossvalidate2(labels, input_shape, data=None, valfrac=0.2, outdir=None, **params):
    #loop through the folds, training on 80% of Nfolds-1, testing on 20% of Nfolds-1,
    #and predicting the remaining fold.
    #valfrac sets the 80/20 split
    params = {'runlabel':'alldata_b_',
                'Nfolds': 5,
                'debug': False,
                'epochs': 400,
                'crop': 512,
                'predicttwoparams': [0, 1],
                'patience': 20,
                'learning_rate': 0.1,
                'decay': True}
    
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    istart = 0
    for fold, [train_index, test_index] in enumerate(kf.split(np.arange(len(labels)))):
        print('regressing fold', fold+1, 'of', 5, '...')
        if outdir is None:
            log_dir = os.path.join(outputdir, 'output', params['runlabel'], str(fold))
        else:
            log_dir = os.path.join(outdir, str(fold))
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

        #fit the model
        print('fitting model...')
        model.fit(data[train_index], labels[train_index], batch_size=32, verbose=2,
        validation_split=0.2, epochs=params['epochs'], callbacks=[cb])
        testData = data[test_index]

        #print out a summary stat:
        print('calculating test loss...')
        score = model.evaluate(testData, labels[test_index], batch_size=32, verbose=1)

        print('          Test loss:', score)
        print('')

        #print('Test loss:', score[0]) #for model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
        #print('Test accuracy:', score[1])
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

        # test resetting model
        K.clear_session()




def plotRandomData(data, labels, Nplot=7, **params):
    print('labels dim:', labels.ndim, labels.shape)
    if labels.shape[1] == 1:
        ind = np.argsort(labels.flatten())
    else:
        #sort by the 2nd label
        ind = np.argsort(labels[:,1])
    numDat = len(ind)
    #plot up the 25th, 50th, and 75th percentile in the label
    indToPlot = [ind[int(0.25*numDat)], ind[int(0.5*numDat)], ind[int(0.75*numDat)]]
    vmin = np.percentile(data, 5)
    vmax = np.percentile(data, 95)
    for i, snapshotnum in enumerate(indToPlot):
        #print('plotting train data', snapshotnum, '...')
        if i == 0:
            addzlabel = False
        else:
            addzlabel = False
        title = None #r'$z_\mathrm{mid}='+str(labels[snapshotnum])+'$'
        #to do:  fix the title above to work for more than one label per image
        plotOneDataSet(data[snapshotnum], labels[snapshotnum], snapshotnum, title,
                       vmin, vmax, addzlabel = addzlabel, **params)

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

def plotFilters(fold=0, noise_type = 'white', step = 1, **params):
    """

    Plot up the interesting filters to interpret the CNN

    noise_type = 'white' gives random noise.  It's the only choice right now, 
    but you can imagine other noises that might be interesting.
    
    step = 1 defines a step size for each iteration

    """


    #load the model and print some information about it
    model = keras.models.load_model('/home/ntampaka/pylon5/21cm/models/fold_'+str(fold)+'.model')
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    model.summary()
    

    #load one example, just to get the geometry of the input images right :)
    data, one_image_shape = readImages([0], **params)
    
    #the previous line failed because I had the wrong file path, so here's a hard-coded version:
    #data = 100*np.ones((1, 512, 512, 30), 'f')
    data_shape = data.shape
    print(data.shape, data_shape)
    #print(data_shape)
    
    #There's probably a more clever or descriptive way to do this that sets a redshift-by-redshift
    #maximum to account for the brightness gradient.

    #initialize some 
    num_successful_filters = 0
    kept_filters = []
    input_img = model.input
    #caution:  if you change the layer_output to something else, you'll also need to change the 
    #max_num_filters by hand, I don't know how to do that automatically.
    #last layer:
    #layer_output = layer_dict['global_average_pooling2d_1'].output
    #max_num_filters = 64
    #trial layer
    layer_output = layer_dict['global_average_pooling2d_1'].output
    max_num_filters = 64
    
    for filter_index in range(max_num_filters):
        print('filter_index', filter_index)

        #caution:  I've hard-coded this for channels_last.
        loss = K.mean(layer_output[:, filter_index])

        # we compute the gradient of the input picture wrt this loss
        #for K.mean, this will just maximize the "brightness" of this filter.
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this is the heart of this.  We figure out how we need to change the input image 
        # to maximize the loss function.  
        iterate = K.function([input_img], [loss, grads])


        # first filters I sent Paul had L2norm=0.8, gauss_filter_sig=5, num_asc_steps=5, step=1
        # we start from a gray image with some random noise
        if noise_type == 'white':
            #input is white noise (good for picking up on textures)
            #if you don't
            input_img_data = np.zeros(data_shape)
            for n in range(data_shape[3]):
                pixmin = 1.2*np.min(data[0,:,:,n])
                pixmax = 1.2*np.max(data[0,:,:,n])
                #print(n, pixmin, pixmax, data_shape, data_shape[1:3])
                input_img_data[0,:,:,n] = (pixmax-pixmin)*np.random.random(data_shape[1:3])+pixmin
            L2norm = 0.8  #how will we penalize very bright pixels?
            gauss_filter_sig = 5  #how many pixels will the gaussian blur reach?

        #plot up the noise, just for talks or whatever :)
        #if filter_index == 0 and fold == 0:
        #    print('data_shape:', data_shape)
        #    for n in range(data_shape[3]):
        ##        plt.imshow(input_img_data[0, :,:,n], cmap='plasma', interpolation = 'nearest')
        #        plt.axis('off')
        #        plt.savefig('noiseexample_'+noise_type+'_'+str(n)+'.pdf')
        #        plt.clf()

        #run gradient ascent for num_asc_steps
        num_asc_steps = 10
        for i in range(num_asc_steps):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step
            #L2 decay
            input_img_data *= L2norm
            #and a gaussian blur to give it that creepy DeepDream feel :P
            for n in range(data_shape[3]):
                input_img_data[0,:,:,n] = gaussian_filter(input_img_data[0,:,:,n], sigma=gauss_filter_sig)

            #if loss_value <= 0:
            #    # some filters get stuck to 0, we can skip them
            #    print(filter_index, 'broke at asc step', i)
            #    break

            #print(i, grads_value.shape, input_img_data.shape)
                    
       
        if loss_value > 0:
            num_successful_filters += 1
            #img = deprocess_image(input_img_data[0])
            kept_filters.append((input_img_data, loss_value))

    dat = np.zeros((num_successful_filters, data_shape[1], data_shape[2], data_shape[3]), 'f')
    for n in range(len(kept_filters)):
        dat[n] = kept_filters[n][0]
    np.save('filters_'+str(fold)+'.npy', dat)

    print('number of successful filters:', num_successful_filters)
    

    makeFilterPlot = False
    if makeFilterPlot == True:
        # next, I'm going to stitch the best filters on a grid.  I've somewhat arbitrarily chosen to 
        # use 3 redshift snapshots (the 10th, 15th, and 20th), which will be the columns.  
        # the successful filters will be the rows
        zind = [3, 9, 10, 11, 12, 16]

    
        # keep the 5 best filters (if there are 5)
        n  = min(num_successful_filters, 8)

        # sort filters by loss (these are probably better, or at least better-looking)
        kept_filters.sort(key=lambda x: x[1], reverse=True)

        # build a black picture with enough space for
        margin = 5
    
        img_width  = data_shape[1]
        img_height = data_shape[2]
        width  = len(zind) * img_width + (len(zind)- 1) * margin
        height = n * img_height + (n - 1) * margin
    
        stitched_filters = np.zeros((width, height))

        print('size checks:', width, height, img_width, img_height)
   
    
        # fill the picture with our saved filters
        for j in range(n):
            img, loss = kept_filters[j]
            for i, z in enumerate(zind):
                stitched_filters[(img_width  + margin) * i: (img_width  + margin) * i + img_width,
                             (img_height + margin) * j: (img_height + margin) * j + img_height] =\
                                                                    img[0,:,:,z]

        # print('pixel range check:', np.min(stitched_filters), np.max(stitched_filters))

        # save the image
        print('stitched filters:', np.max(stitched_filters), np.min(stitched_filters), \
                  stitched_filters.shape)
            

        plt.imshow(stitched_filters, cmap='plasma', interpolation = 'nearest')
        plt.axis('off')
        plt.savefig('filters_'+str(fold)+'.png',\
                bbox_inches = 'tight', dpi=1000)
        plt.clf()







def main(unused_argv):
    #start a timer
    t0 = time.clock()

    for fold in range(5):
        plotFilters(fold = fold)
        print('finished fold', fold)
    sys.exit("Intentionally killed the job here so I can test just the new function :)")
    

    #make a directory for output
    out_prefix = os.path.basename(os.getcwd())
    outdir = os.path.join(outputdir, 'output', out_prefix, params['runlabel'])
    logdir = os.path.join(outputdir, 'output', out_prefix, 'logs')
    ensureDirectory(outdir)
    ensureDirectory(logdir)

    #output text to a file
    logfile = os.path.join(logdir, params['runlabel'] + '.txt')
    #sys.stdout = open(logfile, 'w+', 1)

    print('parameters:', params)

    print('reading labels...')
    if 'ntrain' in params:
        ind = list(range(params['ntrain']))
        labels = readLabels(ind=ind, **params)
    else:
        labels = readLabels(**params)

    print('reading all data ...')
    ind = np.arange(labels.shape[0])
    data, input_shape = readImages(ind=ind, **params)
    print('finished reading data')
    # plot up some of the data:
    # plotRandomData(data, labels, **params)


    #crossvalidate with N folds:
    print('crossvalidating...')
    crossvalidate(labels, input_shape, data=data, outdir=outdir, **params)

    #quick plot of the results
    plotPreds(outdir=outdir)


    print('finished running in ', end='')
    print((time.clock()-t0)/60., end='')
    print(' minutes.')


fold0_model_name = "/pylon5/as5phnp/tbilling/sandbox/fold_0.model"
model = keras.models.load_model(fold0_model_name)
model.get_losses_for()




results = np.load("/pylon5/as5phnp/tbilling/sandbox/results.npy")
truth,prediction,fold = list(zip(*results.tolist()))
truth =np.array(list(zip(*results.tolist())))[:,:,0]
prediction =np.array(list(zip(*results.tolist())))[:,:,1]
plt.figure()
for i in np.arange(1000):
    plt.plot(truth[0,i],prediction[0,i],'k.')
plt.xlabel("Truth")
plt.ylabel("Prediction")
plt.show()
plt.savefig("truth_pred_fold0.png")


if __name__ == "__main__":
    tf.app.run()
