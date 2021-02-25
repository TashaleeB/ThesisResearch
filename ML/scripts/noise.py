import numpy as np, h5py
import os
#import cv2

var_nomode = [.2, .02, .002]
var_mode = [0.01, 0.001, 1e-4 ]

"""
Parameters
----------
image : ndarray
    Input image data. Will be converted to float.
mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
"""

def noisy(noise_typ,image, var):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
  elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
  elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)
      noisy = image + image * gauss
      return noisy

for wedge in [False, True]:
    if not wedge:
        print("Full Data")
        variances = var_nomode
    if wedge:
        print("Wedge Cut Data")
        variances = var_mode
        
    for variance in variances:
        print("Variance ",variance)
        os.system("cp /pylon5/as5phnp/tbilling/data/v12_nowedge.h5 /pylon5/as5phnp/tbilling/data/v12_nowedge_copy.h5")
        os.system("cp /pylon5/as5phnp/tbilling/data/v12_wedge.h5 /pylon5/as5phnp/tbilling/data/v12_wedge_copy.h5")
        data_path = "/pylon5/as5phnp/tbilling/data/"

        if wedge == False:
            inputFile = data_path+'v12_nowedge_copy.h5'
            # Load data and add noise to the training image and test image
            print("Reading in data ...")
            x_test = h5py.File(inputFile, 'r+')['test_images']
            y_test = h5py.File(inputFile, 'r+')['test_labels']
            x_train = h5py.File(inputFile, 'r+')['train_images']
            y_train = h5py.File(inputFile, 'r+')['train_labels']
            
            for i in range(x_test.shape[0]):
                # replace each image with noisy data
                x_test[i] = noisy(noise_typ="gauss",image= x_test[i], var=variance)
                
            for i in range(x_train.shape[0]):
                # replace each image with noisy data
                x_train[i] = noisy(noise_typ="gauss",image= x_train[i], var=variance)
              
            # save the data with noise added to it.
            print("Saving data ...")
            hf = h5py.File(data_path+'v12_nowedge_noise_{}.h5'.format(variance), 'w')
            hf.create_dataset('test_images', data=x_test)
            hf.create_dataset('test_labels', data=y_test)
            hf.create_dataset('train_images', data=x_train)
            hf.create_dataset('train_labels', data=y_train)
            hf.close()
            
        if wedge == True:
            inputFile = data_path+'v12_wedge_copy.h5'
            # Load data and add noise to the training image and test image
            print("Reading in data ...")
            x_test = h5py.File(inputFile, 'r+')['test_images']
            y_test = h5py.File(inputFile, 'r+')['test_labels']
            x_train = h5py.File(inputFile, 'r+')['train_images']
            y_train = h5py.File(inputFile, 'r+')['train_labels']

            for i in range(x_test.shape[0]):
                # replace each image with noisy data
                x_test[i] = noisy(noise_typ="gauss",image= x_test[i], var=variance)
                
            for i in range(x_train.shape[0]):
                # replace each image with noisy data
                x_train[i] = noisy(noise_typ="gauss",image= x_train[i], var=variance)
              
            # save the data with noise added to it.
            print("Saving data ...")
            hf = h5py.File(data_path+'v12_wedge_noise_{}.h5'.format(variance), 'w')
            hf.create_dataset('test_images', data=x_test)
            hf.create_dataset('test_labels', data=y_test)
            hf.create_dataset('train_images', data=x_train)
            hf.create_dataset('train_labels', data=y_train)
            hf.close()

