# needs to be ran in hp_opt environment with Tensorflow version
# https://towardsdatascience.com/bayesian-neural-networks-with-tensorflow-probability-fbce27d6ef6
# Data set: http://archive.ics.uci.edu/ml/datasets/Air+Quality

# tf.__version__ : '2.1.0'

from __future__ import print_function, division, absolute_import

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.datasets import mnist
from tensorflow.keras import Model
#from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Softmax, ReLU, Flatten #Flatten, Activation
from tensorflow.keras.utils import normalize, to_categorical

import gc, time
gc.enable()

tfk = tf.keras

N_EPOCH = 10
BATCH_SIZE = 32
VERBOSE = 1
N_CLASSES = 10
OPTIMIZER = 'adam'
N_HIDDEN = 128
VALIDATION_SPLIT = 0.1 #10%
DROPOUT = 0.5 #50%

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

"""
# Build Sequential Model with Dropout
model_dropout = Sequential()

model_dropout.add(Flatten(input_shape=(28, 28)))
model_dropout.add(Dense(N_HIDDEN))

model_dropout.add(Dropout(DROPOUT))
model_dropout.add(Activation('relu'))
model_dropout.add(Dense(N_HIDDEN))
model_dropout.add(Dropout(DROPOUT))

model_dropout.add(Activation('relu'))
model_dropout.add(Dense(N_CLASSES))
model_dropout.add(Activation('softmax'))

model_dropout.summary()
"""
inputs = Flatten(input_shape=(28, 28))
#inputs = Input(shape=(X_train.shape[1]),)
#inter = Dense(N_CLASSES)(inputs)
inter = Dropout(DROPOUT)(inputs, training=True)
#inter = ReLU()
inter = Dense(N_CLASSES, activation='relu')(inter)
inter = Dropout(DROPOUT)(inter, training=True)
inter = Dense(128, activation='relu')(inter)
output = Dense(N_CLASSES, activation='softmax')(inter)

model_dropout = Model(inputs=inputs, outputs=output)

# We’re trying to predict classes, we use categorical crossentropy as our loss function
# Compile model
model_dropout.compile(
    loss='categorical_crossentropy',
    optimizer=OPTIMIZER,
    metrics=['accuracy'])

start_time = time.time()
# Start Training
history_dropout = model_dropout.fit(
    X_train,
    y_train,
    epochs=N_EPOCH,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    verbose = 1,
    shuffle=True)

running_time = time.time() - start_time

test_loss, test_acc = model_dropout.evaluate(X_test, y_test)

inputs = tfk.Input(shape=(10,))
x = tfk.layers.Dense(3)(inputs)
outputs = tfk.layers.Dropout(0.5)(x, training=True)

model = keras.Model(inputs, outputs)
