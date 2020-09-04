# needs to be ran in hp_opt environment with Tensorflow version
#  https://stackoverflow.com/questions/58566096/custom-loss-function-that-updates-at-each-step-via-gradient-descent

# tf.__version__ : '2.1.0'

from __future__ import print_function, division, absolute_import

import numpy as np, matplotlib.pyplot as plt, gc, time, h5py, keras
import tensorflow as tf

from datetime import timedelta

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Layer, Lambda
from tensorflow.keras.initializers import RandomNormal, RandomUniform
from keras import backend as K
from sklearn.metrics import accuracy_score

from matplotlib.ticker import PercentFormatter

gc.enable()

 # As you are trying to use function decorator in TF 2.0, please enable run function eagerly by using below line after importing
# TensorFlow: https://www.tensorflow.org/guide/effective_tf2#use_tfconfigexperimental_run_functions_eagerly_when_debugging
tf.config.experimental_run_functions_eagerly(True)

wedge = False # Is the data wedge filtered
training = False # if True the dropout will be active during to testing processes
data_path = "/lustre/aoc/projects/hera/tbilling/ml/data/"

if wedge == False:
    inputFile = data_path+"t21_snapshots_nowedge_v9.hdf5"

if wedge == True:
    inputFile = data_path+"t21_snapshots_wedge_v9.hdf5"

outputdir = "/lustre/aoc/projects/hera/tbilling/ml/likelihood/"

train_test_file = data_path+"train_test_index_80_20_split.npz"

n=0
N_EPOCH = 400
factor =1000.

# define a layer for storing the values that go into our loss parameters
class TrainableLossLayer(Layer):
    """
    A class for storing parameters that go into our loss function.

    This class is needed for storing values that are updated during training. We
    define several "required" methods on it so that it can be inserted into a
    keras model properly.

    Parameters
    ----------
    initializer : keras initializer object
        An initializer for generating the initial values of the parameters.

    Returns
    -------
    Layer
        A keras Layer that is suitable for input to a keras graph.
    """
    def __init__(self, initializer, **kwargs):
        super(TrainableLossLayer, self).__init__(**kwargs)
        self.initializer = initializer

    def build(self, input_shape):
        """
        Define what goes into this layer.

        Parameters
        ----------
        input_shape : tuple
            The input shape that should be used for this layer. This is a
            required parameter for a keras layer, even though we're not going to
            explicitly use it in our case.

        Returns
        -------
        None
        """
        self.kernel = self.add_weight(
            name="kernel", shape=(1,), initializer=self.initializer, trainable=True
        )
        self.built = True

    def call(self, inputs):
        """
        Define what this layer should do when it's in a graph.

        In our case, this will just return the value that's currently stored in
        the weight.

        Parameters
        ----------
        inputs : keras layer
            The input that this layer takes. Again, this is a required parameter
            for a custom keras layer that we won't explicitly use.

        Returns
        -------
        keras tensor
            The value that this layer provides in the computational graph.
        """
        return self.kernel

    def compute_output_shape(self, input_shape):
        """
        Define what the output shape of this layer is.

        Because this only puts out a single value (the current value of the
        weight), it's always (1,).

        Parameters
        ----------
        input_shape : tuple
            The input shape. Ignored dummy parameter.

        Returns
        -------
        tuple
            The shape of the output from this layer.
        """
        return (1,)

    def get_config(self):
        """
        Define what the configuration is for this layer.

        When we go to save the model, we need to communicate to keras that there
        is a required parameter that the __init__ function takes. Otherwise it
        will complain and not save our model.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        config = super().get_config().copy()
        config.update(
            {"initializer": self.initializer}
        )

def lambda_loss(x):
    """
    Define our loss function.

    This function will be called as a lambda layer, so that we can pass in
    multiple values to it.

    Parameters
    ----------
    x : tuple of input data
     Our input arguments, which will be unpacked.

    Returns
    -------
    keras tensor
     The value of the loss function.
    """
    y_true, y_pred, mu, sigma = x
    likelihood = -(y_true - y_pred - mu)**2 / (y_true * sigma)**2 - K.log(2 * np.pi * (y_true * sigma)**2)
    # take negative log of likelihood, since we're *minimizing* the loss
    return -1.0 * likelihood

def dummy_loss(y_true, y_pred):
    """
    Define a dummy loss function.

    Parameters
    ----------
    y_true : keras tensor
     The "true" values of my input data.
    y_pred : keras tensor
     The "predicted" values of my network.

    Returns
    -------
    keras tensor
     The "loss" values.
    """
    return y_pred

def readLabels(ind=None, **params):

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

    print('reading data from', inputFile)

    f = h5py.File(inputFile, 'r')

    if 'crop' in params:
        #use just the top corner of the images
        data = np.asarray(f['Data'][u't21_snapshots'][ind, :, 0 : params['crop'], 0 : params['crop']])
    else:
        #use everything!
        print('reading all data', len(ind))
        data = np.asarray(f['Data'][u't21_snapshots'][ind, :, :, :]) # (N_realizations, N_redshifts, N_pix, N_pix)
        #print('loaded data', len(ind))

    print('finished loading data.', data.shape)

    data  = data.transpose(0,2,3,1)  # (N_realizations, N_pix, N_pix, N_redshifts)

    return data, data[0].shape


# Load Index Label
train_index = np.load(train_test_file)["train_index"]
test_index = np.load(train_test_file)["test_index"]

# Load images and labels for training and testing
train_labels = readLabels(ind=None)[train_index,5]*factor
train_labels = train_labels.reshape(-1, 1)
train_images,shape =readImages(ind=train_index)

test_labels = readLabels(ind=None)[test_index,5]*factor
testl_abels = test_labels.reshape(-1, 1)
test_image,input_shape = readImages(ind=test_index)

def model():
    input0 = Input(shape=input_shape)
    y_input = Input(shape=(1,))

    # Now, we need to define initializers for our mu and sigma values.
    # Mu can be positive or negative, so we use a default "normal" value.
    # Sigma must be positive, so we start it out there.
    mu_initializer = RandomNormal(mean=0.0, stddev=1.0)
    sigma_initializer = RandomUniform(minval=1.0, maxval=10.0)

    # Now, we actually *make* the mu and sigma layers we'll need.
    # The input layer we pass in doesn't matter, because it'll be ignored.
    mu = TrainableLossLayer(mu_initializer, name="mu_value")(input0)
    sigma = TrainableLossLayer(sigma_initializer, name="sigma_value")(input0)

    inner = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu')(input0)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)

    inner = Conv2D(32, kernel_size=(3, 3), activation='relu')(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)

    inner = Conv2D(64, kernel_size=(3, 3), activation='relu')(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)

    if wedge == False:
        inner = Conv2D(256, kernel_size=(3, 3), activation='relu')(inner)
        inner = BatchNormalization()(inner)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)

        inner = GlobalAveragePooling2D()(inner)

        inner = Dropout(0.2)(inner, training=True)
        inner = Dense(350, activation='relu')(inner)

    else:
        inner = Conv2D(128, kernel_size=(3, 3), activation='relu')(inner)
        inner = BatchNormalization()(inner)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)

        inner = Conv2D(128, kernel_size=(3, 3), activation='relu')(inner)
        inner = BatchNormalization()(inner)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)

        inner = GlobalAveragePooling2D()(inner)

        inner = Dropout(0.2)(inner, training=True)
        inner = Dense(250, activation='relu')(inner)

    inner = Dropout(0.2)(inner, training=True)
    inner = Dense(200, activation='relu')(inner)

    inner = Dropout(0.2)(inner, training=True)
    inner = Dense(100, activation='relu')(inner)

    inner = Dropout(0.2)(inner, training=True)
    inner = Dense(20, activation='relu')(inner)

    output = Dense(1)(inner)

    # The "output" value of the network is in the "d3" layer. So that's the "y_pred"
    # that I'm going to use in my loss function. So I'm going to call a Lambda layer
    # with the right arguments now, which will be my loss function.
    loss = Lambda(lambda_loss)([y_input, output, mu, sigma])

    model_dropout = Model(inputs=[input0,y_input], outputs=loss)

    # Compile Model
    model_dropout.compile(loss=dummy_loss,optimizer=keras.optimizers.Adam(lr=0.01, decay=0.))

    # Summary of model used
    print(model_dropout.summary())

    return model_dropout
    

start_time = time.time()

# Finally, our model needs fake input data to do back propagation, and isn't
# smart enough to know that it doesn't actually need it. So we just make zeros
# to fake it out.
dummy_train = np.zeros_like(train_labels)
#dummy_test = np.zeros_like(test_labels)

# Start Training
model_dropout = model()
#history_dropout = model_dropout.fit(train_images, train_labels, epochs=N_EPOCH,
#                                    batch_size=32, validation_split=0.1, verbose = 2, shuffle=True)
history_dropout = model_dropout.fit([train_images[:720], train_labels[:720]], dummy_train[:720],
     validation_data=([train_images[720:], train_labels[720:]],
     dummy_train[720:]), batch_size=32,
     epochs=N_EPOCH, verbose = 2, shuffle=True)

running_time = time.time() - start_time
print("Finish Training CNN in ", str(timedelta(seconds=running_time)))

# Save model information
if wedge == True:
    # Save model and weights
    print("saving model trained on wedge filtered data ...")
    model_dropout.save(outputdir+"likelihood_CNN_model_wedge.h5")
    model_dropout.save_weights(outputdir+"likelihood_CNN_weights_wedge.h5")

    # Save history
    print("Removing Scaling factor ({}) and saving histories...".format(factor))
    history_keys = np.array(list(history_dropout.history.keys()))
    for key in history_keys:
        np.savez(outputdir+"likelihood_CNN_wedge_history",
                  metric=np.array(history_dropout.history[str(key)])/factor)

if wedge == False:
    # Save model and weights
    print("saving model trained on nowedge filtered data ...")
    model_dropout.save(outputdir+"likelihood_CNN_model_nowedge.h5")
    model_dropout.save_weights(outputdir+"likelihood_CNN_weights_nowedge.h5")

    # Save history
    print("Removing Scaling factor ({}) and saving histories...".format(factor))
    history_keys = np.array(list(history_dropout.history.keys()))
    for key in history_keys:
        np.savez(outputdir+"likelihood_CNN_nowedge_history",
              metric=np.array(history_dropout.history[str(key)])/factor)

# evaluate trained model
test_loss = model_dropout.evaluate(test_image, test_labels)

# make predictions
dropout_predictions = []
for i in range(500):
    y_p = model_dropout.predict(test_image, batch_size=test_labels.shape[0])
    dropout_predictions.append(y_p) # (500, 100, 1) = (# of masks, # of datasets, # of classes)

# select an index from the 200 prediciton over 500 dropout masks
idx = 50
p0 = np.array([p[idx] for p in dropout_predictions])
print("posterior mean: {}".format(p0.mean(axis=0)))
print("true label: {}".format(test_labels[idx]/factor))
print()

# probability and variance
for i, (prob, var) in enumerate(zip(p0.mean(axis=0), p0.std(axis=0))):
    print("class: {}; probability: {:.1%}; var: {:.2%} ".format(i, prob, var))
    
# Plot a 2D histogram ???? https://matplotlib.org/3.1.1/gallery/statistics/hist.html
fig, ax = plt.subplots(tight_layout=True)
hist = ax.hist2d(x, y, bins=10)

# look at the Probability distributions of the monte carlo predictions and in blue you see the prediction of the ensemble
plt.figure(figsize=(12,12))
plt.hist(p0[:,i], bins=100, density=True)
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
ax.set_title(f"class {i}")
ax.label_outer()

# one-to-one with errorbars (Fig 11) https://arxiv.org/pdf/1911.08508.pdf
results = glob.glob(outputdir+"*.npy")
result=np.load(results[0])

# Convert to true tau units
true_tau = low_z_tau + h_2 * result["truth"][:,n]/factor
predicted_tau = low_z_tau + h_2 * result["prediction"][:,n]/factor

plt.figure(figsize=(12,12))
plt.errorbar(true_tau, predicted_tau, xerr=xerr, yerr=yerr, fmt='-o')
#plt.scatter(true_tau, predicted_tau, s=6, lw=0, alpha=0.9, label=fold[r])
x = np.linspace(0.95*np.min(true_tau), 1.05*np.max(true_tau), 1000)
plt.plot(x, x, 'k--',lw=1,alpha=0.2)
plt.xlabel()
plt.ylabel()
