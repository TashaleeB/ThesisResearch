import os, h5py
# Use scikit-learn to grid search the batch size and epochs
import numpy as np
import matplotlib.pyplot as plt

data_path = '/pylon5/as5phnp/tbilling/data/'
inputFile = data_path+'t21_snapshots_nowedge_v7.hdf5'

def readLabels(ind=None, **params):
    """
        read in labels only
        (to use with batches
    """
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

labels = readLabels(ind=None)

def z_25(_zmid, _delta_z):
    z25 = _zmid + _delta_z/2.
    
    return z25

def z_75(_zmid, _delta_z):
    z75 = _zmid - _delta_z/2.
    
    return z75

labels = readLabels(ind=None)
z_25 = z_25(_zmid=labels[[],0], _delta_z=labels[[],1])
z_75 = z_75(_zmid=labels[[],0], _delta_z=labels[[],1])


plt.figure(figsize=(15,10))
plt.plot(labels[0,0],labels[0,5], 'k.')
plt.plot(labels[0,1],labels[0,5], 'k.')
plt.plot(labels[0,0],labels[0,5], 'k.')

