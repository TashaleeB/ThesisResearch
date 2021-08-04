# needs to be ran in hp_opt environment with Tensorflow version
# https://keras.io/examples/vision/visualizing_what_convnets_learn/
# https://www.machinecurve.com/index.php/2019/11/18/visualizing-keras-model-inputs-with-activation-maximization/
# Data set: MNIST

# tf.__version__ : '2.1.0'

from __future__ import print_function, division, absolute_import

import numpy as np, matplotlib.pyplot as plt, gc, time, h5py, keras, os
import tensorflow as tf

from datetime import timedelta
from IPython.display import Image, display

from tensorflow.keras import Model
from tensorflow.keras.models import load_model

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from matplotlib.ticker import PercentFormatter

gc.enable()

# As you are trying to use function decorator in TF 2.0, please enable run function eagerly by using below line after importing TensorFlow: https://www.tensorflow.org/guide/effective_tf2#use_tfconfigexperimental_run_functions_eagerly_when_debugging
tf.config.experimental_run_functions_eagerly(True)

wedge = False # Is the data wedge filtered
data_path = "/ocean/projects/ast180004p/tbilling/data/"

if wedge == False:
    inputFile = data_path+"t21_snapshots_nowedge_v12.hdf5"
    perfectmodel = "/ocean/projects/ast180004p/tbilling/sandbox/redo_mlpaper/no_modes_removed/CNN_model_nowedge_1.h5" # nowedge

if wedge == True:
    inputFile = data_path+"t21_snapshots_wedge_v12.hdf5"
    perfectmodel = "/ocean/projects/ast180004p/tbilling/sandbox/redo_mlpaper/modes_removed/CNN_model_wedge_5.h5" # wedge
    
outputdir = "/ocean/projects/ast180004p/tbilling/sandbox/bayesian/denseflipout/sandbox/load_ml_p1_model_stream_v12Data_DenseVar_layer_end/"

# The dimensions of our input image
img_width = 512
img_height = 512

train_test_file = data_path + "train_test_index_80_20_split.npz"

n=0
batch_size = 32
N_EPOCH = 200
factor =1000.


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
    # Create a custom loss function that divides the difference by the true
    #if not K.is_tensor(y_pred):
    #if not K.is_keras_tensor(y_pred):
    #    y_pred = K.constant(y_pred)

    y_true = K.cast(y_true, y_pred.dtype) #Casts a tensor to a different dtype and returns it.
    diff_ratio = K.square((y_pred - y_true)/K.clip(K.abs(y_true),K.epsilon(),None))
    loss = K.mean(diff_ratio, axis=-1)
    # Return a function

    return loss

# Load training data
training_labels = readLabels(ind=None)[np.arange(800),5]*factor
training_labels = training_labels.reshape(-1, 1)
training_images,input_shape = readImages(ind=np.arange(800))

# Build a feature extraction model
model = load_model(perfectmodel,
custom_objects={"Mean_Squared_over_true_Error": Mean_Squared_over_true_Error})
print(model.summary())

# Our target layer: we will visualize the filters from this layer.
# See `model.summary()` for list of layer names, if you want to change this.
layer_name = model.layers[10].name
print("*"*50)
print("LAYER NAME: ",layer_name)
print("*"*50)
# Set up a model that returns the activation values for our target layer
layer = model.get_layer(name=layer_name)
feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)
    
    
#Set up the gradient ascent process
#The "loss" we will maximize is simply the mean of the activation of a specific filter in our target layer. To avoid border effects, we exclude border pixels.
def compute_loss(input_image, filter_index):
    activation = feature_extractor(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)
    
    
# Our gradient ascent function simply computes the gradients of the loss above with regard to the input image, and update the update image so as to move it towards a state that will activate the target filter more strongly.
@tf.function
def gradient_ascent_step(img, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img
    
    
#Set up the end-to-end filter visualization loop
#Our process is as follow:
#Start from a random image that is close to "all gray" (i.e. visually netural)
#Repeatedly apply the gradient ascent step function defined above
#Convert the resulting input image back to a displayable form, by normalizing it, center-cropping it, and restricting it to the [0, 512] range.

def initialize_image():
    # We start from a gray image with some random noise
    img = tf.random.uniform((1, img_width, img_height, 30))
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return (img - 0.5) * 0.25


def visualize_filter(filter_index):
    # We run gradient ascent for 20 steps
    iterations = 30
    learning_rate = 10.0
    img = initialize_image()
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate)

    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())
    return loss, img


def deprocess_image(img):
    # https://becominghuman.ai/image-data-pre-processing-for-neural-networks-498289068258
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    #img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    #img += 0.5
    #img = np.clip(img, 0, 1)

    # Convert to RGB array
    #img *= img_width
    #img = np.clip(img, 0, img_width).astype("uint8")
    return img


#Let's try it out with filter 0 in the target layer:
loss, img = visualize_filter(0)
#keras.preprocessing.image.save_img("0.png", img) # does not work! ugh why??

# Visualize the first 64 filters in the target layer
# Now, let's make a 8x8 grid of the first 64 filters in the target layer to get of feel for the range of different visual patterns that the model has learned.
all_imgs = []
for filter_index in range(100):
    print("Processing filter %d" % (filter_index,))
    loss, img = visualize_filter(filter_index)
    all_imgs.append(img)
    
np.savez(outputdir+"activation_max_CNN_wedge_filters", images=np.array(all_imgs))

# Build a black picture with enough space for
# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
margin = 5
n = 8
cropped_width = img_width #- 25 * 2
cropped_height = img_height #- 25 * 2
width = n * (cropped_height + margin)
height = n * (cropped_height + margin)
stitched_filters = np.zeros((width, height, 30))

# Fill the picture with our saved filters
for i in range(n):
    for j in range(n):
        img = all_imgs[i * n + j]
        stitched_filters[
            (cropped_width + margin) * i : (cropped_width + margin) * i + cropped_width,
            (cropped_height + margin) * j : (cropped_height + margin) * j
            + cropped_height,
            :,
        ] = img
  
np.savez(outputdir+"activation_max_CNN_wedge_stictched_filters", images=stitched_filters)

for i in range(30):
    plt.figure()
    plt.imshow(stitched_filters[:,:,i], cmap= "seismic")
    plt.colorbar()
    plt.show()

"""
# Save model information
if wedge == True:
    # Save model
    print("saving model trained on wedge filtered data ...")
    model_dropout.save_weights(outputdir+"amax_CNN_weights_wedge.h5")
    model_dropout.save(outputdir+"amax_CNN_model_wedge.h5")
    
    # Save history
    print("Removing Scaling factor ({}) and saving histories...".format(factor))
    history_keys = np.array(list(history_dropout.history.keys()))
    for key in history_keys:
        np.savez(outputdir+"amax_CNN_wedge_history",
                 metric=np.array(history_dropout.history[str(key)])/factor)
                 
if wedge == False:
    # Save model
    print("saving model trained on nowedge filtered data ...")
    model_dropout.save_weights(outputdir+"amax_CNN_weights_nowedge.h5")
    model_dropout.save(outputdir+"amax_CNN_model_nowedge.h5")

    # Save history
    print("Removing Scaling factor ({}) and saving histories...".format(factor))
    history_keys = np.array(list(history_dropout.history.keys()))
    for key in history_keys:
        np.savez(outputdir+"amax_CNN_nowedge_history",
                 metric=np.array(history_dropout.history[str(key)])/factor)

# evaluate trained model
test_loss = model_dropout.evaluate(test_images, test_labels)

# make predictions
dropout_predictions = []
for i in range(500):
    y_p = model_dropout.predict(test_images, batch_size=test_labels.shape[0])
    dropout_predictions.append(y_p) # (500, 100, 1) = (# of masks, # of datasets, # of classes)

# Save dropout predictions
if wedge == False:
    np.savez(outputdir+"dropout_CNN_nowedge_pedictions", prediciton = dropout_predictions)
if wedge == True:
    np.savez(outputdir+"dropout_CNN_wedge_pedictions", prediciton = dropout_predictions)

# select an index from the 200 prediciton over 500 dropout masks
idx = 50
p0 = np.array([p[idx] for p in dropout_predictions])
print("posterior mean: {}".format(p0.mean(axis=0)))
print("true label: {}".format(test_labels[idx]/factor))
print()

# probability and variance
for i, (prob, var) in enumerate(zip(p0.mean(axis=0), p0.std(axis=0))):
    print("class: {}; probability: {:.1%}; var: {:.2%} ".format(i, prob, var))
    
# ???? Plot a 2D histogram ???? https://matplotlib.org/3.1.1/gallery/statistics/hist.html
fig, ax = plt.subplots(tight_layout=True)
hist = ax.hist2d(x, y, bins=10)

# look at the Probability distributions of the monte carlo predictions and in blue you see the prediction of the ensemble
plt.figure(figsize=(12,12))
plt.hist(p0[:,i], bins=100, density=True)
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
if wedge == True:
    plt.savefig(outputdir+"MC_PDF_wedge.png")
if wedge == False:
plt.savefig(outputdir+"MC_PDF_nowedge.png")

# one-to-one with errorbars (Fig 11) https://arxiv.org/pdf/1911.08508.pdf
results = glob.glob(outputdir+"*.npy")
result=np.load(results[0])
yerr =

# Convert to true tau units
h_2 = 0.45321170409999995
low_z_tau = 0.030029479627917934
true_tau = low_z_tau + h_2 * result["truth"][:,n]/factor
predicted_tau = low_z_tau + h_2 * result["prediction"][:,n]/factor

plt.figure(figsize=(12,12))
plt.errorbar(true_tau, predicted_tau, xerr=xerr, yerr=yerr, fmt='-o')
#plt.scatter(true_tau, predicted_tau, s=6, lw=0, alpha=0.9, label=fold[r])
x = np.linspace(0.95*np.min(true_tau), 1.05*np.max(true_tau), 1000)
plt.plot(x, x, 'k--',lw=1,alpha=0.2)
plt.xlabel()
plt.ylabel()
"""


"""
...: factor = 1000
...: result = np.mean(np.load("nowedge/predictions.npz")["prediciton"][:,:,0]/factor, axis=0)
...: yerr = np.std(np.load("nowedge/predictions.npz")["prediciton"][:,:,0]/factor, axis=0)
...:
...: # Convert to true tau units
...: h_2 = 0.45321170409999995
...: low_z_tau = 0.030029479627917934
...: true_tau = low_z_tau + h_2 * result
...: predicted_tau = low_z_tau + h_2 * test_labels/factor
...:
...:
...: plt.figure(figsize=(12,12))
...: plt.errorbar(true_tau, predicted_tau, yerr=yerr, fmt='o')
...: #plt.errorbar(true_tau, predicted_tau, xerr=xerr, yerr=yerr, fmt='-o')
...: #plt.scatter(true_tau, predicted_tau, s=6, lw=0, alpha=0.9, label=fold[r])
...: x = np.linspace(0.95*np.min(true_tau), 1.05*np.max(true_tau), 1000)
...: plt.plot(x, x, 'k--',lw=3,alpha=0.2)
...: plt.xlabel("True", fontsize=16)
...: plt.ylabel("Predicted", fontsize=16)
...: plt.savefig("nowedge/predictions.png")
"""
