# needs to be ran in hp_opt environment with Tensorflow version
"""
 https://stackoverflow.com/questions/50124158/keras-loss-function-with-additional-dynamic-parameter

OR THIS IS ANOTHER OPTION

https://stackoverflow.com/questions/45961428/make-a-custom-loss-function-in-keras/45963039#45963039
"""

# Data set: MNIST

# tf.__version__ : '2.1.0'

from __future__ import print_function, division, absolute_import

import numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from tensorflow.keras.datasets import mnist
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Softmax, ReLU, Flatten, Dropout
from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score

import gc, time
gc.enable()

N_EPOCH = 30
BATCH_SIZE = 32
VERBOSE = 1
N_CLASSES = 10
OPTIMIZER = 'adam'
N_HIDDEN = 128
VALIDATION_SPLIT = 0.1 #10%
DROPOUT = 0.5 #50%

# Normalize MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build model
# manually input arrays
inputs0 = Input(shape=(28, 28))
# Hidden Layeres
inputs1 = Flatten()(inputs0)
inter = Dropout(rate=DROPOUT)(inputs1, training=True)
inter = tfp.layers.DenseFlipout(units=N_HIDDEN, activation='relu')(inter)
output1 = tfp.layers.DenseFlipout(units=N_CLASSES + 1, activation='softmax')(inter)
output = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1],
scale=1e-3 + tf.math.softplus(0.01 * t[...,1:])))(output1)

# Total params: 101,770     Total params: 203,659
# Trainable params: 101,770     Trainable params: 203,659


model_flipout = Model(inputs=inputs0, outputs=output)

# We’re trying to predict classes, we use categorical crossentropy as our loss function
# Compile model
model_flipout.compile(
    loss="categorical_crossentropy",
    optimizer=OPTIMIZER,
    metrics=['accuracy'])

start_time = time.time()
# Start Training
history_flipout = model_flipout.fit(
    X_train,
    y_train,
    epochs=N_EPOCH,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    verbose = 1,
    shuffle=True)

running_time = time.time() - start_time

model_flipout(X_test)
#<tfp.distributions.Normal 'model_distribution_lambda_Normal' batch_shape=[10000, 10] event_shape=[] dtype=float32>

# evaluate trained model
test_loss, test_acc = model_flipout.evaluate(X_test, y_test)

# make predictions
# Make a testing model (which shares all weights in training model)
test_model = Model( inputs=inputs0, outputs=output, name='test_only' )

flipout_predictions = []
for i in range(500):
    y_p = test_model.predict(X_test, batch_size=1000)
    flipout_predictions.append(y_p) #(500, 10000, 10) = (# of masks, # of datasets, # of classes)

# calculate mean accuracy over distibution of acc for each mask for a trained network
accs = []
for y_p in flipout_predictions:
    # for each mask return the max value along the class axis
    acc = accuracy_score(y_test.argmax(axis=1), y_p.argmax(axis=1), normalize=True) # return the fraction of correctly classified samples.
    accs.append(acc) # (500,)
# should not be better than history_accu/test_acc ???
print("Flipout accuracy: {:.1%}".format(sum(accs)/len(accs))) # (1,)

# score ensemble of the dropout mask
# find mean along the mask axis and return max value along the class axis
flipout_ensemble_pred = np.array(flipout_predictions).mean(axis=0).argmax(axis=1) #(10000,)
ensemble_acc = accuracy_score(y_test.argmax(axis=1), flipout_ensemble_pred, normalize=True) #return the fraction of correctly classified samples.
print("Flipout-ensemble accuracy: {:.1%}".format(ensemble_acc))

# look at the distributions of the monte carlo predictions and in blue you see the prediction of the ensemble
plt.figure()
#plot distibution of the accuracy
plt.hist(accs, color="r")
plt.axvline(x=history_flipout.history['accuracy'][-1], color="c")
plt.axvline(x=history_flipout.history['val_accuracy'][-1], color="g")
plt.axvline(x=sum(accs)/len(accs), color="k")
plt.axvline(x=ensemble_acc, color="b")

# select an index from the 1000 prediciton over 500 dropout masks
idx = 247
p0 = np.array([p[idx] for p in flipout_predictions])
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









# create model
model = Sequential()

# add layers
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))





def dice_coef(y_true, y_pred,is_weight):
    value = is_weight*categorical_crossentropy( y_true, y_pred )

    return value

def dice_loss(is_weight):
  def dice(y_true, y_pred):
    return -dice_coef(y_true, y_pred,is_weight)
  return dice

inputs0 = Input(shape=(28, 28))
is_weight0 = Input(shape=(1,))

# Hidden Layeres
inputs1 = Flatten()(inputs0)
inter = Dense(128, activation='relu')(inputs1)
output = Dense(10, activation='softmax')(inter)

model = Model(inputs=[inputs0, is_weight0], outputs=output)
#model.add_loss(dice_loss(is_weight=0.5))
model.compile(optimizer="adam",loss=dice_loss(is_weight=0.05))
#model.compile( loss=None, optimizer='adam' )
print(model.summary())
model.fit(x=X_train, y=y_train)
