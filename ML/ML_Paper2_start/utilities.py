"""
This script has some miscellaneous untility functins that are used throughout the training process.
"""

# Import libraries
from __future__ import print_function, division, absolute_import
import numpy as np
import os, sys, h5py
import tensorflow_probability as tfp, tensorflow as tf

tfd = tfp.distributions

from keras_tuner.tuners import RandomSearch
from tensorflow import keras
from keras import backend as K
from tqdm import tqdm

def does_project_name_exist(project_name):
    # check to see if path exists
    if os.path.isdir(project_name):
        os.system("rm -rvf {}".format(project_name))
        #print("{} exists and will be deleted...".format(project_name))
        
    return

def crop(image, img_height, img_width, crop_length):
    
    start_y = (img_height - crop_length) // 2
    start_x = (img_width - crop_length) // 2
    cropped_image=image[:, start_x:(img_width - start_x), start_y:(img_height - start_y), :]
    
    return cropped_image

def save_model_modelweights(bcnn_model, path, filter, training_data_name, model_name, loss_name):

    # check to see if dir exists and if not create it
    MYDIR = os.path.join(path, filter)
    CHECK_FOLDER = os.path.isdir(MYDIR)
    
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("Created folder : ", MYDIR)

    else:
        print(MYDIR, "Folder already exists.")
    
    # save model and model weights
    print("Saving best model and model weights...")
    bcnn_model.save(os.path.join(path, filter,"best_model_{}_{}_{}.h5".format(training_data_name, model_name, loss_name)))
    bcnn_model.save_weights(os.path.join(path, filter,"best_model_weights_{}_{}_{}.h5".format(training_data_name, model_name, loss_name)))
    
def readLabels(inputFile, ind=None, **params):
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

def readImages(inputFile, ind, **params):
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

def load_zreion_data(inputFile, factor = 1000):
    """
    inputFile : str
        Location of HDF5 file. This is the data made by Paul.
    FACTOR : int
        Scaling factor used to prevent numerical underflow.
    """
    
    # "/ocean/projects/ast180004p/tbilling/data/t21_snapshots_nowedge_v12.hdf5"
    trainlabels = readLabels(inputFile, ind=None)[:,5]*factor
    trainlabels = trainlabels.reshape(-1, 1)
    training_data,shape =readImages(inputFile, ind=np.arange(1000))
    
    nx, ny, _ = shape
    X_train = training_data[:800]
    y_train = trainlabels[:800,:]
    X_test = training_data[800:]
    y_test = trainlabels[800:,:]
    
    
    return nx, ny, training_data, X_train, y_train, X_test, y_test


def load_data(training_data_name, path, file_name, factor = 100):
    """
    Parameters
    ----------
    TRAINING_DATA_NAME : str
        The name simulation of the model used to train and test the CNN.
        eg. "toymodel" or "zreion" or "21cmfast"
    PATH : str
        Model to train.
    FILE_NAME : str
        Name of the data file
        "toy_models_512x512.npz" or "GenLight_1sim.hdf5"
    """

    if training_data_name.lower() == 'toymodel':
        print("Loading Toy Model data... ")
        # load data
        toy_model = np.load('toy_models_32x32.npz')#np.load('toy_models_32x32.npz')
        nx, ny, ntrain = toy_model['training_data'].shape
        training_data = toy_model['training_data'].T
        labels = toy_model['labels']

        # Normalizing the data
        X_train = training_data[0:8000,:,:].reshape(8000,nx,ny,1)
        y_train = labels[0:8000]*factor
        X_test = training_data[8000:,:,:].reshape(2000,nx,ny,1)
        y_test = labels[8000:]*factor

        print("training", X_train.shape)
        print("validation", X_test.shape)
        
    if training_data_name.lower() == '21cmfast':
        # if 21cmfast or zreioon data load  the data this way instead.
        d = h5py.File(file_name, 'r')
        for key in d.keys():
            print(key, d[key][()].shape)
            
        nx, ny, ntrain = d['lightcone_brightness_temperature_21'][()][:,:,0::10].shape

        training_data = d['lightcone_brightness_temperature_21'][()][:,:,0::10][..., np.newaxis]
        labels = d['lightcone_xHI_21'][()][:,:,0::10].mean(axis=(0,1))
        
        # create random indicies
        np.random.seed(420)
        random_idx = np.arange(ntrain)
        np.random.shuffle(random_idx)
        print("Shuffled indicies", random_idx)

        # Normalizing the data
        X_train = training_data[:,:,random_idx[0:180],:].reshape(180,nx,ny,1)
        y_train = labels[random_idx[0:180]]*factor
        X_test = training_data[:,:,random_idx[180:],:].reshape(23,nx,ny,1)
        y_test = labels[random_idx[180:]]*factor
                
    if training_data_name.lower() == 'zreion':
        # if 21cmfast or zreioon data load  the data this way instead.
        d = h5py.File(file_name, 'r')
        for key in d.keys():
            print(key, d[key][()].shape)
            
        nx, ny, ntrain = d['lightcone_brightness_temperature_zre'][()][:,:,0::10].shape

        training_data = d['lightcone_brightness_temperature_zre'][()][:,:,0::10][..., np.newaxis]
        labels = d['lightcone_xHI_zre'][()][:,:,0::10].mean(axis=(0,1))

        # create random indicies
        np.random.seed(420)
        random_idx = np.arange(ntrain)
        np.random.shuffle(random_idx)
        print("Shuffled indicies", random_idx)

        # Normalizing the data
        X_train = training_data[:,:,random_idx[0:180],:].reshape(180,nx,ny,1)
        y_train = labels[random_idx[0:180]]*factor
        X_test = training_data[:,:,random_idx[180:],:].reshape(23,nx,ny,1)
        y_test = labels[random_idx[180:]]*factor
        
    return nx, ny, training_data, X_train, y_train, X_test, y_test


# Normalization term for divergence
def kl_divergence_normalization_term(TRAINING_DATA):
    """
    Parameters
    ----------
    TRAINING_DATA : ndarray
        Input image data. Will be converted to int
    """

    NUM_TRAIN_EXAMPLES = len(TRAINING_DATA)
    kl_divergence_function = (
    lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(NUM_TRAIN_EXAMPLES,dtype=tf.float32)
    )
    return kl_divergence_function

def random_search(training_data_name, model,num_models,outputdir,project_name, X_train, y_train, X_test, y_test, epochs, batch_size, objective='mse'):
    """
    Parameters
    ----------
    TRAINING_DATA_NAME : str
        The name simulation of the model used to train and test the CNN.
        eg. "ToyModel" or "zreion" or "21cmfast"
    MODEL_NAME : str
        some creative name for the model you just train.
        eg. "model1_prob_loss" or "model1_nll_loss"
    MODEL : keras.engine.sequential.Sequential
        Model to train.
    NUM_MODELS : int
        Max number of training models
    OUTPUTDIR : str
        Output directory to save model
    PROJECT_NAME : str
        Name of the project to save yperparameter training.
    X_TRAIN : ndarray
        Input image data before the CNN is trained.
    Y_TRAIN : ndarray
        Image labels for the CNN while it is being trained.
    X_TEST : ndarray
        Input image data after the CNN is trained it is tested with this dataset.
    Y_TEST : ndarray
        Image labels. After the CNN is trained it is tested with this dataset.
    EPOCH : int
        Number of training steps. The number of times to go through the full training data.
    BATCH_SIZE : int
        Number of training data required to update the model.
    OBJECTIVE : str
        The type oof metric to minimize. This helps the tuner deteremine the best possible model.
    LOSS_NAME : str
        The prob. loss function used duing the training process.
        eg. "elbo" or "nll"
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"
    """
    
    # Check to see if project exists before you start trainging
    #does_project_name_exist(project_name)
    
    # Start Tuning based on low validation loss
    tuner = RandomSearch(model,
                         objective=objective, # 'loss', 'val_loss', 'val_accuracy'
                         max_trials= num_models,# specify the number of different models to try
                         executions_per_trial=1, #2,
                         directory=outputdir,
                         seed=42,
                         project_name=project_name)

    # Print a summary of the search space
    tuner.search_space_summary()

    # Show the best models, their hyperparameters, and the resulting metrics.
    tuner.search(X_train, y_train,
                 epochs=epochs,#hp.Int('epoch',min_value=10,max_value=HP_EPOCH,step=10,default=10),
                 validation_data=(X_test, y_test),
                 batch_size=batch_size,
                #callbacks=[plotlosses],
                verbose=1)
                
    print("-"*50)
    print("Tuning Summary")
    print("-"*50)
    tuner.results_summary()
    tuner.search_space_summary()

    # Retrieve the best model.
    best_model = tuner.get_best_models(num_models=1)[0]
    
    # check to see if dir exists and if not create it
    MYDIR = os.path.join(path, filter)
    CHECK_FOLDER = os.path.isdir(MYDIR)
    
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("Created folder : ", MYDIR)

    else:
        print(MYDIR, "Folder already exists.")
    
    # save model and model weights
    print("Saving best model and model weights...")
    best_model.save(os.path.join(path, filter,"best_model_{}_{}.npz".format(training_data_name, model_name)))
    best_model.save_weights(os.path.join(path, filter,"best_model_weights_{}_{}.npz".format(training_data_name, model_name)))
    

    return tuner.get_best_models(num_models=20)
    
    

def mc_pred(training_data_name, bestmodel, X_test, y_test, model_name, path, filter="nowedge"):
    # make list of predictions using the best model
    """
    Parameters
    ----------
    TRAINING_DATA_NAME : str
        The name simulation of the model used to train and test the CNN.
        eg. "ToyModel" or "zreion" or "21cmfast"
    X_TEST : ndarray
        Input image data. Will be converted to int. Length 1.
    BESTMODEL : keras.engine.sequential.Sequential
        Best model determined from kerastuner.
    MODEL_NAME : str
        some creative name for the model you just train.
        eg. "model1_prob_loss" or "model1_nll_loss"
    PATH : str
        output path
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"

    """
    pred = []

    for i in tqdm(range(500)):
        y_p = bestmodel.predict(X_test).squeeze()#predict(X_test, batch_size=test_labels.shape[0])
        pred.append(y_p)
    distrib_predictions = np.array(pred)
    # check to see if dir exists and if not create it
    MYDIR = os.path.join(path, filter)
    CHECK_FOLDER = os.path.isdir(MYDIR)
    
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("Created folder : ", MYDIR)

    else:
        print(MYDIR, "Folder already exists.")
    
    # save predictions
    print("Saving predictions...")
    np.savez(os.path.join(path, filter,"mc_pred_{}_{}.npz".format(training_data_name, model_name)), predictions= distrib_predictions, labels=y_test)

    return distrib_predictions

def mc_pred_tuner_models(training_data_name, tuner_models, X_test, y_test, model_name, path, filter="nowedge"):
    # make list of predictions using all of the model
    """
    Parameters
    ----------
    TRAINING_DATA_NAME : str
        The name simulation of the model used to train and test the CNN.
        eg. "ToyModel" or "zreion" or "21cmfast"
    X_TEST : ndarray
        Input image data. Will be converted to int. Length 1.
    TUNER_MODELS : keras.engine.sequential.Sequential
        List of model determined from kerastuner.
    MODEL_NAME : str
        some creative name for the model you just train.
        eg. "model1_prob_loss" or "model1_nll_loss"
    PATH : str
        output path
    FILTER : str
        is this data wedge filtered or not.
        eg. "nowedge" or "wedge"

    """
    # make predictions
    prediction = []
    mean_predictions = []#np.zeros([len(X_test),])
    distrib_predictions = []#np.zeros([len(X_test),])

    for m in tuner_models:
        for i in tqdm(range(500)):
            y_p = m.predict(X_test).squeeze()#predict(X_test, batch_size=test_labels.shape[0])
            prediction.append(y_p) # (500, 2000) == (num mc sample, num test images)
        distrib_predictions.append(prediction) # (500, 4, 2000) == (num mc sample, num models, num test images)
        prediction = np.mean(np.array(prediction), axis=0) # (, 2000) == (, num test images)
        mean_predictions.append(prediction) # (4, 2000) == (num models, num test images)
        prediction = []
    distrib_predictions = np.array(distrib_predictions)
    mean_predictions = np.array(mean_predictions)
    # check to see if dir exists and if not create it
    MYDIR = os.path.join(path, filter)
    CHECK_FOLDER = os.path.isdir(MYDIR)
    
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("Created folder : ", MYDIR)

    else:
        print(MYDIR, "Folder already exists.")
    
    # save predictions
    print("Saving predictions...")
    np.savez(os.path.join(path, filter,"mc_pred_tuner_models_{}_{}.npz".format(training_data_name, model_name)), distrib_predictions= distrib_predictions, mean_predictions=mean_predictions, labels=y_test)
    
    print(X_test.shape)
    print("Mean prredictions:", np.array(mean_predictions).shape)
    print("Distibution predictioons:", np.array(distrib_predictions))

    
    return np.array(mean_predictions), np.array(distrib_predictions)
    

def add_noise_to_data(variances, X_train, X_test):
    """
    Parameters
    ----------
    X_TRAIN : ndarray
        Input image data to train CNN.
    X_TEST : ndarray
        Input image data to test CNN.
    VARIANCE: list of float
        There are different levels of nose.
        eg. var_nomode = [.01, .001, .0001]
            var_mode = [0.01, 0.001, 1e-4 ]
    """

    def noisy(noise_typ, image, var):

        """
        Parameters
        ----------
        IMAGE : ndarray
            Input image data.
        NOISE_TYPE : str
            One of the following strings, selecting the type of noise to add:

            'gauss'     Gaussian-distributed additive noise.
            'poisson'   Poisson-distributed noise generated from the data.
            's&p'       Replaces random pixels with 0 or 1.
            'speckle'   Multiplicative noise using out = image + n*image,where
                        n is uniform noise with specified mean & variance.
        VAR: float
            There are different levels of nose.
            eg. .01 or .001 or .0001 or 1e-4
        """

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


    noisy_X_train = []
    noisy_X_test = []
    for variance in variances:
        print("Variance ",variance)
        # copying over data
        print("Reading in data ...")
        x_test = copy.deepcopy(X_test)
        x_train = copy.deepcopy(X_train)

        for i in range(x_test.shape[0]):
            # replace each image with noisy data
            x_test[i] = noisy(noise_typ="gauss",image= x_test[i], var=variance)

        for i in range(x_train.shape[0]):
            # replace each image with noisy data
            x_train[i] = noisy(noise_typ="gauss",image= x_train[i], var=variance)

        # append to list
        print("Append data ...")
        noisy_X_train.append(x_train)
        noisy_X_test.append(x_test)
           
    noisy_X_train = np.array(noisy_X_train)
    noisy_X_test = np.array(noisy_X_test)
    
    print("Training data shape:", noisy_X_train.shape)
    print("Test data shape:", noisy_X_test.shape)
    
    return noisy_X_train, noisy_X_test




class custom_loss:

    """
    Parameters
    ----------
    Y_TRUE : ndarray
        Input array dataset. Will be converted to float.
    Y_PRED:
        Image labels. Will be converted to float.
    """

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    # Custom Loss Functions
    kl_divergence = tf.keras.losses.KLDivergence()

    neg_log_likelihood = lambda y_true, y_pred: -tf.reduce_mean(input_tensor=y_pred.log_prob(y_true))

    def Mean_Squared_over_true_Error(y_true, y_pred):
        # Create a custom loss function that divides the difference by the true

        y_true = K.cast(y_true, y_pred.dtype) #Casts a tensor to a different dtype and returns it.
        diff_ratio = K.square((y_pred - y_true)/K.clip(K.abs(y_true),K.epsilon(),None))

        loss = K.mean(diff_ratio, axis=-1)

        # Return a tensor
        return loss

    def elbo(y_true, y_pred):
        kl_weight = 1
        neg_log_likelihood = -tf.reduce_mean(input_tensor=y_pred.log_prob(y_true))
        kl_divergence = tf.keras.losses.KLDivergence()

        elbo_loss = -tf.math.reduce_mean(-kl_weight * kl_divergence(y_true, y_pred.mean()) - neg_log_likelihood)
        # Return a tensor
        return elbo_loss

    def mean_fractional_error(y_true, y_pred):
        y_true = K.cast(y_true, y_pred.dtype) #Casts a tensor to a different dtype and returns it.
        diff_ratio = (y_pred - y_true)/K.clip(K.abs(y_true),K.epsilon(),None)
        loss = K.mean(diff_ratio, axis=-1)
        
        return loss
        
    def r2_keras(y_true, y_pred):
        SS_res =  K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )

    keras.utils.get_custom_objects().update({'Mean_Squared_over_true_Error':Mean_Squared_over_true_Error,
                                            'neg_log_likelihood': neg_log_likelihood, 'kl_divergence':kl_divergence,
                                            'elbo':elbo,
                                            'r2_keras':r2_keras
                                            })
                                            
