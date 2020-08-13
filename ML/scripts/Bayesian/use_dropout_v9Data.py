# needs to be ran in hp_opt environment with Tensorflow version
# https://www.depends-on-the-definition.com/model-uncertainty-in-deep-learning-with-monte-carlo-dropout/
# Data set: MNIST

# tf.__version__ : '2.1.0'

from __future__ import print_function, division, absolute_import

import numpy as np, matplotlib.pyplot as plt, gc, time
import tensorflow as tf

from datetime import timedelta

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.utils import normalize
from sklearn.metrics import accuracy_score

gc.enable()

data_path = '/pylon5/as5phnp/tbilling/data/'
wedge = False # Is the data wedge filtered

if wedge = False:
    inputFile = data_path+'t21_snapshots_nowedge_v7.hdf5'

if wedge = True:
    inputFile = data_path+'t21_snapshots_wedge_v7.hdf5'
    
outputdir = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/"

train_test_file = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/train_test_index_0.npz"

N_EPOCH = 200
factor =1000.


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

# Load Index Label
train_index = np.load(train_test_file)["train_index"]
test_index = np.load(train_test_file)["test_index"]

# Load images and labels
train_labels = readLabels(ind=None)[train_index,5]*factor
train_labels = train_labels.reshape(-1, 1)
train_images,shape =readImages(ind=train_index)

test_labels = readLabels(ind=None)[test_index,5]*factor
testl_abels = test_labels.reshape(-1, 1)
test_image,input_shape = readImages(ind=test_index)

def model(image_shape):
    input0 = Input(shape=image_shape)
    #inner = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu',input_shape=images.shape[1:])(input0)
    inner = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu')(input0)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    
    inner = Conv2D(32, kernel_size=(3, 3), activation='relu')(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    
    inner = Conv2D(64, kernel_size=(3, 3), activation='relu')(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)
    
    if wedge = False:
        inner = Conv2D(256, kernel_size=(3, 3), activation='relu')(inner)
        inner = BatchNormalization()(inner)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)

        inner = GlobalAveragePooling2D()(inner)
        
        inner = Dropout(0.2)(inner)
        inner = Dense(350, activation='relu')(inner)
    
    else:
        inner = Conv2D(128, kernel_size=(3, 3), activation='relu')(inner)
        inner = BatchNormalization()(inner)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)
        
        inner = Conv2D(128, kernel_size=(3, 3), activation='relu')(inner)
        inner = BatchNormalization()(inner)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)

        inner = GlobalAveragePooling2D()(inner)

        inner = Dropout(0.2)(inner)
        inner = Dense(250, activation='relu')(inner)
    
    inner = Dropout(0.2)(inner)
    inner = Dense(200, activation='relu')(inner)
    
    inner = Dropout(0.2))
    inner = Dense(100, activation='relu')(inner)
    
    inner = Dropout(0.2)(inner)
    inner = Dense(20, activation='relu')(inner)
    
    output = Dense(1)(inner)
    
    model_dropout = Model(inputs=inputs0, outputs=output)
    
    # Compile Model
    model_dropout.compile(loss=Mean_Squared_over_true_Error,optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.))

    # Summary of model used
    print(model_dropout.summary())
    
    return model_dropout
    

start_time = time.time()
# Start Training
model_dropout = model(image_shape=shape)
history_dropout = model_dropout.fit(
    train_images,
    train_labels,
    epochs=N_EPOCH,
    batch_size=32,
    validation_split=0.1,
    verbose = 1,
    shuffle=True)

running_time = time.time() - start_time
print("Finish Training NN ", running_time)
        
# Save model information
if wedge = True:
    # Save model
    print("saving model trained on wedge filtered data ...")
    model_dropout.save(outputdir+"model_wedge.h5")
    
    # Save history
    print("Removing Scaling factor ({}) and saving histories...".format(factor))
    history_keys = np.array(list(history_dropout.history.keys()))
    for key in history_keys:
        np.savez(outputdir+"wedge_history",
                 metric=np.array(history_dropout.history[str(key)])/factor)
                 
if wedge = False:
    # Save model
    print("saving model trained on nowedge filtered data ...")
    model_dropout.save(outputdir+"model_nowedge.h5")

    # Save history
    print("Removing Scaling factor ({}) and saving histories...".format(factor))
    history_keys = np.array(list(history_dropout.history.keys()))
    for key in history_keys:
        np.savez(outputdir+"nowedge_history",
                 metric=np.array(history_dropout.history[str(key)])/factor)

# evaluate trained model
test_loss, test_acc = model_dropout.evaluate(test_images, test_labels)

# make predictions
dropout_predictions = []
for i in range(500):
    y_p = model_dropout.predict(test_images, batch_size=test_labels.shape[0])
    dropout_predictions.append(y_p) # (500, 200, 1) = (# of masks, # of datasets, # of classes)

# calculate mean accuracy over distibution of acc for each mask for a trained network
accs = []
for y_p in dropout_predictions:
    # for each mask return the max value along the class axis
    acc = accuracy_score(test_labels.argmax(axis=1), y_p.argmax(axis=1), normalize=True) # return the fraction of correctly classified samples.
    accs.append(acc) # (500,)
# should not be better than history_accu/test_acc ???
print("Dropout accuracy: {:.1%}".format(sum(accs)/len(accs))) # (1,)

# score ensemble of the dropout mask
# find mean along the mask axis and return max value along the class axis
dropout_ensemble_pred = np.array(dropout_predictions).mean(axis=0).argmax(axis=1) #(200,)
ensemble_acc = accuracy_score(test_labels.argmax(axis=1), dropout_ensemble_pred, normalize=True) #return the fraction of correctly classified samples.
print("Dropout-ensemble accuracy: {:.1%}".format(ensemble_acc))

# look at the distributions of the monte carlo predictions and in blue you see the prediction of the ensemble
plt.figure()
#plot distibution of the accuracy
plt.hist(accs, color="r")
plt.axvline(x=history_dropout.history['accuracy'][-1], color="c")
plt.axvline(x=history_dropout.history['val_accuracy'][-1], color="g")
plt.axvline(x=sum(accs)/len(accs), color="k")
plt.axvline(x=ensemble_acc, color="b")

# select an index from the 200 prediciton over 500 dropout masks
idx = 50
p0 = np.array([p[idx] for p in dropout_predictions])
print("posterior mean: {}".format(p0.mean(axis=0).argmax()))
print("true label: {}".format(y_test[idx].argmax()))
print()

# probability and variance
for i, (prob, var) in enumerate(zip(p0.mean(axis=0), p0.std(axis=0))):
    print("class: {}; probability: {:.1%}; var: {:.2%} ".format(i, prob, var))
    
# Probability ditribution plot
fig, axes = plt.subplots(5, 2, figsize=(12,12))
for i, ax in enumerate(fig.get_axes()):
    ax.hist(p0[:,i], bins=100, range=(0,1));
    ax.set_title(f"class {i}")
    ax.label_outer()

