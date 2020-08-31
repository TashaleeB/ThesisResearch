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

from tensorflow.keras.datasets import mnist
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Softmax, ReLU, Flatten
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

# Custom Loss Function
def sample_loss(y_true, output, sigma, mu):
    return sigma * mu * categorical_crossentropy( y_true, output )

# Build model
# manually input arrays
inputs0 = Input(shape=(28, 28))
#y_true = Input( shape=(10,), name='y_true' )
sigma = Input(shape=(1,), name='std')
mu = Input(shape=(1,), name='mean')

# Hidden Layeres
inputs1 = Flatten()(inputs0)
inter = Dropout(DROPOUT)(inputs1, training=True)
inter = Dense(N_HIDDEN, activation='relu')(inter)
output = Dense(N_CLASSES, activation='softmax')(inter)

# Since loss was added through the add_loss() function, we don't need too go through compile( loss=xxx ).
#model_dropout = Model(inputs=[inputs0, y_true, sigma, mu], outputs=output)
model_dropout = Model(inputs=[inputs0, sigma, mu], outputs=output)
#model_dropout.add_loss(sample_loss(y_true, output, sigma, mu))
model_dropout.add_loss(sample_loss(inputs0, output, sigma, mu))

# We’re trying to predict classes, we use categorical crossentropy as our loss function
# Compile model
model_dropout.compile(
    loss=None,
    optimizer=OPTIMIZER,
    metrics=['accuracy'])

# We need to initialize these new parameters
sigma0 = np.random.rand()#np.random.randn(8,32,32,3)
mu0 = np.random.rand()#np.random.randn(8,10)
#np.random.randint(0,2,size=(8,1))

start_time = time.time()
# Start Training
history_dropout = model_dropout.fit(
    [X_train, sigma0, mu0],
    y_train,
    epochs=N_EPOCH,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    verbose = 1,
    shuffle=True)

running_time = time.time() - start_time

# evaluate trained model
test_loss, test_acc = model_dropout.evaluate(X_test, y_test)

# make predictions
# Make a testing model (which shares all weights in training model)
test_model = Model( inputs=inputs0, outputs=output, name='test_only' )

dropout_predictions = []
for i in range(500):
    y_p = test_model.predict(X_test, batch_size=1000)
    dropout_predictions.append(y_p) #(500, 10000, 10) = (# of masks, # of datasets, # of classes)

# calculate mean accuracy over distibution of acc for each mask for a trained network
accs = []
for y_p in dropout_predictions:
    # for each mask return the max value along the class axis
    acc = accuracy_score(y_test.argmax(axis=1), y_p.argmax(axis=1), normalize=True) # return the fraction of correctly classified samples.
    accs.append(acc) # (500,)
# should not be better than history_accu/test_acc ???
print("Dropout accuracy: {:.1%}".format(sum(accs)/len(accs))) # (1,)

# score ensemble of the dropout mask
# find mean along the mask axis and return max value along the class axis
dropout_ensemble_pred = np.array(dropout_predictions).mean(axis=0).argmax(axis=1) #(10000,)
ensemble_acc = accuracy_score(y_test.argmax(axis=1), dropout_ensemble_pred, normalize=True) #return the fraction of correctly classified samples.
print("Dropout-ensemble accuracy: {:.1%}".format(ensemble_acc))

# look at the distributions of the monte carlo predictions and in blue you see the prediction of the ensemble
plt.figure()
#plot distibution of the accuracy
plt.hist(accs, color="r")
plt.axvline(x=history_dropout.history['accuracy'][-1], color="c")
plt.axvline(x=history_dropout.history['val_accuracy'][-1], color="g")
plt.axvline(x=sum(accs)/len(accs), color="k")
plt.axvline(x=ensemble_acc, color="b")

# select an index from the 1000 prediciton over 500 dropout masks
idx = 247
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
