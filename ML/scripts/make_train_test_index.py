from __future__ import print_function, division, absolute_import

# Imports
import h5py, random, glob
import numpy as np

from sklearn.model_selection import KFold

#random seed to control the reproducability
seed = 8675309
np.random.seed(seed)
factor = 1000.

data_path = '/pylon5/as5phnp/tbilling/data/'
reionfilename = data_path+'t21_snapshots_nowedge_v9.hdf5'
outputdir = '/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/'
inputFile = reionfilename

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

trainlabels = readLabels(ind=None)[:,5]*factor
trainlabels = trainlabels.reshape(-1, 1)

kf = KFold(n_splits=10, random_state=seed, shuffle=True)
istart = 0
for fold, [train_index, test_index] in enumerate(kf.split(np.arange(len(trainlabels)))):

    np.savez(outputdir+"train_test_index_{}.npz".format(fold),
            train_index=train_index,test_index=test_index)
