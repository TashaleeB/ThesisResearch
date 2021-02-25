import numpy as np
from tensorflow import keras


n=0
batch_size = 32
N_EPOCH = 200
factor =1000.

wedge = False
data_path = "/pylon5/as5phnp/tbilling/data/"
outputdir = "/pylon5/as5phnp/tbilling/sandbox/bayesian/"

train_test_file = data_path + "train_test_index_80_20_split.npz"

if wedge == False:
    inputFile = data_path+"t21_snapshots_nowedge_v12.hdf5"

if wedge == True:
    inputFile = data_path+"t21_snapshots_wedge_v12.hdf5"
    
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

def Mean_Squared_over_true_Error(y_true, y_pred):

    y_true = K.cast(y_true, y_pred.dtype) #Casts a tensor to a different dtype and returns it.
    diff_ratio = K.square((y_pred - y_true)/K.clip(K.abs(y_true),K.epsilon(),None))
    loss = K.mean(diff_ratio, axis=-1)
    # Return a function
    return loss

# Load training data
test_index = np.load(train_test_file)["test_index"]
test_labels = readLabels(ind=None)[test_index,5]*factor
test_labels = test_labels.reshape(-1, 1)
test_images,input_shape = readImages(ind=test_index)

# Load model and make predictions
if wedge == False:
    model_dropout = keras.models.load_model(outputdir+"dopout_CNN_model_nowedge.h5",
    custom_objects={"Mean_Squared_over_true_Error":Mean_Squared_over_true_Error})
    
    dropout_predictions = []
    for j in range(10):
        for i in range(50):
            y_p = model_dropout.predict(test_images, batch_size=test_labels.shape[0])
            dropout_predictions.append(y_p) # (50, 512, 512, 30) = (# of image, # of pixels, # of pixels, # of classes)
        # save predictions
        np.savez(outputdir+"dropout_CNN_nowedge_pedictions_{}".format(str(j+1)), prediciton = np.array(dropout_predictions))
        
if wedge == True:
    model_dropout = keras.models.load_model(outputdir+"dopout_CNN_model_wedge.h5",
    custom_objects={"Mean_Squared_over_true_Error":Mean_Squared_over_true_Error})
    
    dropout_predictions = []
    for j in range(10):
        for i in range(50):
            y_p = model_dropout.predict(test_images, batch_size=test_labels.shape[0])
            dropout_predictions.append(y_p) # (50, 512, 512, 30) = (# of image, # of pixels, # of pixels, # of classes)
        # save predictions
        np.savez(outputdir+"dropout_CNN_wedge_pedictions_{}".format(str(j+1)), prediciton = dropout_predictions)

