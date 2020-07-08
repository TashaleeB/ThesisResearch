from __future__ import print_function, division, absolute_import

# Imports
import warnings
warnings.filterwarnings('ignore')

import h5py, glob, random
import numpy as np

import gc
gc.enable()

data_path = '/pylon5/as5phnp/tbilling/data/'
outputdir = data_path
inputFile = data_path+'t21_snapshots_nowedge_v9.hdf5'
outputFile_train = outputdir+'modified_training_t21_snapshots_nowedge_v9.hdf5'
outputFile_test = outputdir+'modified_testing_t21_snapshots_nowedge_v9.hdf5'

factor = 1000.

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
"""
number of bytes in a variable in MB
images.nbytes/1e6
"""

images,shape_label = readImages(ind=np.arange(1000)) #~ 31.5 GiB
labels = readLabels(ind=None)[:,5]*factor
labels = labels.reshape(-1, 1)

# Veritically Stack images
img_90 =  np.rot90(images, k=1, axes=(1, 2))
new_image1 = np.vstack((images,img_90))
del(img_90)
gc.collect()

img_180 = np.rot90(images, k=2, axes=(1, 2))
new_image2 = np.vstack((new_image1,img_180))
del(img_180,new_image1)
gc.collect()

img_270 = np.rot90(images, k=3, axes=(1, 2))
new_image3 = np.vstack((new_image2,img_270))
del(img_270,new_image2)
gc.collect()

img_lr = np.flip(images,axis=1)
new_image4 = np.vstack((new_image3,img_lr))
del(img_lr,new_image3)
gc.collect()

img_ud = np.flip(images,axis=0)
new_image5 = np.vstack((new_image4,img_ud))
del(img_ud,new_image4)
gc.collect()

# Veritically Stack Labels
new_labels = np.vstack((labels,labels,labels,labels,labels,labels)) # ~0.024 MB

# Create shuffled indices
min = 0
max = new_labels.shape[0]
digits = np.array([(random.randint(min, max-1)) for i in range(max)])
test_index = digits[:int(len(digits)*0.2)]
train_index = digits[int(len(digits)*0.2):]

# Save training data
#hf = h5py.File(outputFile, 'w')
hf = h5py.File(outputFile_train, 'w')
hf.create_dataset('t21_snapshots', data=new_image5[train_index,:,:,:])
hf.create_dataset('snapshot_labels', data=new_labels[train_index,:])
hf.close()

# Save test data
hf = h5py.File(outputFile_test, 'w')
hf.create_dataset('t21_snapshots', data=new_image5[test_index,:,:,:])
hf.create_dataset('snapshot_labels', data=new_labels[test_index,:])
hf.close()


