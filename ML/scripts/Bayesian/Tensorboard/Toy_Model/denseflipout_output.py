import numpy as np, matplotlib.pyplot as plt, datetime, tensorflow as tf, os, datetime
import tensorflow_probability as tfp

tfd = tfp.distributions

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical, plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
from tensorboard.plugins.hparams import api as hp
#from keras.utils.generic_utils import get_custom_objects

import gc
gc.enable()

# As you are trying to use function decorator in TF 2.0, please enable run function eagerly by using below line after importing TensorFlow: https://www.tensorflow.org/guide/effective_tf2#use_tfconfigexperimental_run_functions_eagerly_when_debugging
#tf.config.experimental_run_functions_eagerly(True)

toy_model = np.load('toy_models.npz')
nx, ny, ntrain = toy_model['training_data'].shape
training_data = toy_model['training_data'].T
labels = toy_model['labels']
outputdir = "/ocean/projects/ast180004p/tbilling/sandbox/bayesian/denseflipout/sandbox/toy_model/denseflipout_output/"

# Data Preprocessing
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 28, 28, 1) / 255.0
X_test = X_test.reshape(10000, 28, 28, 1) / 255.0
X_train = training_data[0:8000,:,:].reshape(8000,nx,ny,1)
y_train = labels[0:8000]
X_test = training_data[8000:,:,:].reshape(2000,nx,ny,1)
y_test = labels[8000:]

print("training", X_train.shape)
print("validation", X_test.shape)


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

"""
https://github.com/qubvel/segmentation_models/issues/374
I changed keras.utils.generic_utils.get_custom_objects().update(custom_objects) to keras.utils.get_custom_objects().update(custom_objects) in .../lib/python3.6/site-packages/efficientnet/__init__.py and it solved the issue.
"""
#get_custom_objects().update({'elbo':elbo}) # this doesn't works
keras.utils.get_custom_objects().update({'Mean_Squared_over_true_Error':Mean_Squared_over_true_Error,
                                        'neg_log_likelihood': neg_log_likelihood, 'kl_divergence':kl_divergence,
                                        'elbo':elbo})

# Experiment setup and the HParams experiment summary
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([10, 20]))
HP_NUM_FILTERS = hp.HParam('num_filters', hp.Discrete([16, 32])) #hp.IntInterval(16, 32))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_LOSS = hp.HParam('loss_fun', hp.Discrete(['mape', 'mse',
                                            'Mean_Squared_over_true_Error', 'neg_log_likelihood',
                                            'kl_divergence', 'elbo']))
HP_EPOCH = hp.HParam('epoch', hp.Discrete([50, 100, 150, 200]) )
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([10, 30, 50, 70, 90, 110]))
HP_LAYER = hp.HParam('layer', hp.Discrete(['Flatten', 'GlobalAveragePooling2D']))
HP_LEARNING_RATE = hp.HParam('lr', hp.Discrete([0.1, 0.01, 0.001, 0.0001]))

METRIC_LOSS = 'loss'
    

with tf.summary.create_file_writer('hparam_logs/hparam_tuning').as_default():
    hp.hparams_config(
    hparams=[HP_NUM_FILTERS, HP_DROPOUT, HP_LOSS, HP_EPOCH, HP_BATCH_SIZE, HP_LAYER, HP_LEARNING_RATE],
    metrics=[hp.Metric(METRIC_LOSS, display_name='Loss')],
    )
  
# Adapt TensorFlow runs to log hyperparameters and metrics
def train_test_model(hparams):
    # Build model
    model = Sequential()
    # Add layers
    model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(nx, ny, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(hparams[HP_NUM_FILTERS], kernel_size=3, activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    if hparams[HP_LAYER] == 'Flatten':
        model.add(Flatten())
    if hparams[HP_LAYER] == 'GlobalAveragePooling2D':
        model.add(GlobalAveragePooling2D())
    
    model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(Dense(200, activation="relu"))
    model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(Dense(10, activation="relu"))
    #model.add(Dense(hparams[HP_NUM_UNITS], activation="relu"))
    #model.add(tfp.layers.DenseFlipout(20, activation="relu"))
    
    model.add(tfp.layers.DenseFlipout(1, activation="linear"))
    #model.add(Dense(1,activation="linear"))
    if hparams[HP_LOSS] == 'neg_log_likelihood' or hparams[HP_LOSS] == 'kl_divergence' or hparams[HP_LOSS] == 'elbo':
        model.add(tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)))
    
    model.compile(optimizer= keras.optimizers.Adam(lr=hparams[HP_LEARNING_RATE], decay=0.), loss = hparams[HP_LOSS]) #hparams[HP_OPTIMIZER], loss="mape")#, metrics=["accuracy"])
    
    # Visualize Model
    print(model.summary())
    plot_model(model, show_shapes=True, show_layer_names=True)
    
    filepath_model = outputdir+"models/flipout_CNN_model_num_units-"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+str(hparams[HP_NUM_UNITS])+"_dropout-"+str(hparams[HP_DROPOUT]) + "_numFilters-"+ str(hparams[HP_NUM_FILTERS])+ "_loss-"+ str(hparams[HP_LOSS])+ "_training_epoch-"+ str(hparams[HP_EPOCH])+ "_batchsize-"+ str(hparams[HP_BATCH_SIZE])+ "_lr-"+ str(hparams[HP_LEARNING_RATE])+ "_layer-"+str(hparams[HP_LAYER])+"-{epoch:02d}-{loss:.4f}.h5"
    filepath_weight = outputdir+"models/flipout_CNN_weight_num_units-"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+str(hparams[HP_NUM_UNITS])+"_dropout-"+str(hparams[HP_DROPOUT]) + "_numFilters-"+ str(hparams[HP_NUM_FILTERS])+ "_loss-"+ str(hparams[HP_LOSS])+ "_training_epoch-"+ str(hparams[HP_EPOCH])+ "_batchsize-"+ str(hparams[HP_BATCH_SIZE])+ "_lr-"+ str(hparams[HP_LEARNING_RATE])+ "_layer-"+str(hparams[HP_LAYER])+"-{epoch:02d}-{loss:.4f}.h5"
    logdir = outputdir+"fit_logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
    hparams_callback = hp.KerasCallback(logdir, hparams)
    
    checkpoint_model = ModelCheckpoint(filepath_model, monitor='loss', verbose=1,
                        save_best_only=True, save_weights_only = False, mode='min', save_freq=5)
    checkpoint_weight = ModelCheckpoint(filepath_weight, monitor='loss', verbose=1,
                        save_best_only=True, save_weights_only = True, mode='min', save_freq=5)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=30)
    callbacks_list = [tensorboard_callback, # log metrics
                        hparams_callback, # log hparams
                        checkpoint_weight, # save copy of model weight
                        checkpoint_model, # save copy of model
                        early_stop] # Stop training when model stops improving
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
                epochs=hparams[HP_EPOCH], batch_size=hparams[HP_BATCH_SIZE],
                callbacks = callbacks_list)
    
    loss = model.evaluate(X_test, y_test)
    return loss

def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        loss = train_test_model(hparams)
        tf.summary.scalar(METRIC_LOSS, loss, step=1)
    
session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value): # because this is an inverval
        for num_filters in HP_NUM_FILTERS.domain.values:
            for loss_fun in HP_LOSS.domain.values:
                for epoch in HP_EPOCH.domain.values:
                    for batch_size in HP_BATCH_SIZE.domain.values:
                        for layer in HP_LAYER.domain.values:
                            for lr in HP_LEARNING_RATE.domain.values:
                                hparams = {
                                  HP_NUM_UNITS: num_units,
                                  HP_DROPOUT: dropout_rate,
                                  HP_NUM_FILTERS: num_filters,
                                  HP_LOSS: loss_fun,
                                  HP_EPOCH: epoch,
                                  HP_BATCH_SIZE: batch_size,
                                  HP_LAYER: layer,
                                  HP_LEARNING_RATE: lr
                                }
                                run_name = "run-%d" % session_num
                                print('--- Starting trial: %s' % run_name)
                                print({h.name: hparams[h] for h in hparams})
                                run('hparam_logs/hparam_tuning/' + run_name, hparams)
                                session_num += 1

# Convert to true tau units
true_tau = low_z_tau + h_2 * result["truth"][:,n]/factor
predicted_tau = low_z_tau + h_2 * result["prediction"][:,n]/factor

# make predictions
predictions = []
for i in range(500):
    y_p = model.predict(X_test).squeeze()#predict(X_test, batch_size=test_labels.shape[0])
    predictions.append(y_p) # (500, 100, 1) = (# of masks, # of datasets, # of classes)
#predictions = model.predict(X_test).squeeze()

plt.plot(y_test, predictions, '.')
plt.plot(y_test,y_test, "r--")
plt.show()




































kl = tf.keras.losses.KLDivergence()
neg_log_likelihood = -tf.reduce_mean(input_tensor=model(X_train[0:1]).log_prob(y_train[0:1]))
neg_log_likelihood.numpy() + kl(y_train[0:1], model(X_train[0:1]).mean()).numpy()
#y_pred.mean()



# Build model
model = Sequential()
# Add layers
model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(nx, ny, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=3, activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(GlobalAveragePooling2D())

model.add(Dropout(0.2))
model.add(Dense(200, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="relu"))
#model.add(Dense(hparams[HP_NUM_UNITS], activation="relu"))
#model.add(tfp.layers.DenseFlipout(20, activation="relu"))

model.add(tfp.layers.DenseFlipout(1, activation="linear"))
#model.add(Dense(1,activation="linear"))

model.add(tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)))

model.compile(optimizer= keras.optimizers.Adam(lr=0.001, decay=0.), loss = tf.keras.losses.KLDivergence())#neg_log_likelihood) #hparams[HP_OPTIMIZER], loss="mape")#, metrics=["accuracy"])

# Visualize Model
print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test))



def elbo(y_true, y_pred):
    kl_weight = 1
    neg_log_likelihood = -tf.reduce_mean(input_tensor=y_pred.log_prob(y_true))
    kl_divergence = tf.keras.losses.KLDivergence()#(y_true, y_pred.mean())#/800 # [kldiv_function / num_examples]

    #elbo_loss = -tf.math.reduce_mean(tf.math.subtract(tf.math.multiply(-kl_weight, kl_divergence), neg_log_likelihood))
    #elbo_loss = -tf.math.reduce_mean(-kl_weight * kl(y_train[0:1], model(X_train[0:1]).mean()).numpy() - neg_log_likelihood.numpy())
    #-tf.math.reduce_mean(tf.math.subtract(tf.math.multiply(-kl_weight,kl_divergence(y_true, y_pred.mean()).numpy()),  neg_log_likelihood.numpy()))
    
    
    #elbo_loss = -tf.math.reduce_mean(-kl_weight * kl_divergence(y_train[0:1], model(X_train[0:1]).mean()) - neg_log_likelihood)
    elbo_loss = -tf.math.reduce_mean(-kl_weight * kl_divergence(y_true, y_pred.mean()) - neg_log_likelihood)
    return elbo_loss

elbo(y_train[0:1], model(X_train[0:1]))
