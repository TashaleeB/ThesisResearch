# needs to be ran in hp_opt environment with Tensorflow version
# tf.__version__ : '2.0.0'

from __future__ import print_function, division, absolute_import

# Imports
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import os, math, numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import gc
gc.enable()

# Load data into memory and split data into training and testing sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

class DataGenerator(tf.compat.v2.keras.utils.Sequence):# General blueprint to make objects
    """
    This data generator will be loading in data from memory batch by batch. Eventually I would like to try to
    this but instead load the data batch by batch from disk.
    """

    def __init__(self, X_data , y_data, batch_size, dim, n_classes, to_fit, shuffle = True): # Constructor
        # Setting the instances(Attribute)
        # This constructor allows you to not have to specify the attributes every single time.
        """Initialization
        :param batch_size = batch_size
        :param X_data = a numpy array containing a batch
        of images with shape (batch_size, *target_size, channels)
        :param labels = list of image labels (file names)
        :param y_data = a numpy array of corresponding labels.
        :param to_fit = True to return X and y, False to return X only
        :param n_classes = number of output
        :param dim = tuple indicating image dimension
        :param shuffle = True to shuffle label indexes after every epoch
        :param n = Batch index. Starts at index (n) 0 and iter until it passes through the entire dataset
        :param list_IDs = list of all 'label' ids to use in the generator
        :param on_epoch_end() = Updates indexes after each epoch
        """
        
        self.batch_size = batch_size
        self.X_data = X_data
        self.labels = y_data
        self.y_data = y_data
        self.to_fit = to_fit
        self.n_classes = n_classes
        self.dim = dim
        self.shuffle = shuffle
        self.n = 0
        self.list_IDs = np.arange(len(self.X_data))
        self.on_epoch_end()
        
    def __next__(self): # Constructor
        # Get one batch of data
        data = self.__getitem__(self.n)
        # Batch index
        self.n += 1

        # If we have processed the entire dataset then
        if self.n >= self.__len__():
            self.on_epoch_end
        self.n = 0

        return data

    def __len__(self): # Constructor
    # Return the number of batches of the dataset
        return math.ceil(len(self.indexes)/self.batch_size)

    def __getitem__(self, index): # Constructor
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X = self._generate_x(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X

    def on_epoch_end(self): #method/attribute
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.X_data))
        if self.shuffle:
            np.random.shuffle(self.indexes)
     
    def _generate_x(self, list_IDs_temp):#method starting with a single underscore is intended for internal use
        # Generates data containing batch_size images
        # Initialization
        X = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self.X_data[ID]

        # Normalize data
        X = (X/255).astype('float32')
        
        return X[:,:,:, np.newaxis]
        
    def _generate_y(self, list_IDs_temp):#method starting with a single underscore is intended for internal use
        y = np.empty(self.batch_size)

        for i, ID in enumerate(list_IDs_temp):
            y[i] = self.y_data[ID]

        return keras.utils.to_categorical(y,num_classes=self.n_classes)


# Build a classification net
n_classes = 10
input_shape = (28, 28)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28, 28 , 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes,activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

# Make an instance of our generators
train_generator = DataGenerator(X_data=x_train, y_data=y_train, batch_size = 64,
                                dim=input_shape,n_classes=10, to_fit=True, shuffle=True)
val_generator = DataGenerator(X_data=x_test, y_data=y_test, batch_size=64, dim = input_shape,
                                n_classes= n_classes, to_fit=True, shuffle=True)

# Call the next() method that yields a batch of samples and labels to check if the generator is working
images, labels = next(train_generator)
print(images.shape)
print(labels.shape)


steps_per_epoch = len(train_generator)
validation_steps = len(val_generator)

# Train the network with the keras function fit_generator()
model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch,
                    epochs=10, validation_data=val_generator,
                    validation_steps=validation_steps)
