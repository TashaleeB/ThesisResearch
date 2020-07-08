# needs to be ran in hp_opt environment with Tensorflow version
# tf.__version__ : '2.0.0'
# https://github.com/gabrielpierobon/cnnshapes/blob/master/README.md

#Let's first import all our required libraries:
​
import glob
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import imageio as im
from keras import models
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

#These are our training images:
#Circles
images = []
for img_path in glob.glob('training_set/circles/*.png'):
    images.append(mpimg.imread(img_path))
​
plt.figure(figsize=(20,10))
columns = 5
for i, image in enumerate(images):
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    plt.imshow(image)

#Squares
images = []
for img_path in glob.glob('training_set/squares/*.png'):
    images.append(mpimg.imread(img_path))
​
plt.figure(figsize=(20,10))
columns = 5
for i, image in enumerate(images):
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    plt.imshow(image)

#Triangles
images = []
for img_path in glob.glob('training_set/triangles/*.png'):
    images.append(mpimg.imread(img_path))
​
plt.figure(figsize=(20,10))
columns = 5
for i, image in enumerate(images):
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    plt.imshow(image)

#The shape of the images:

img = im.imread('training_set/squares/drawing(40).png')
img.shape
(28, 28, 3) Images shapes are of 28 pixels by 28 pixels in RGB scale (although they are arguably black and white only)

#Let's now proceed with our Convolutional Neural Network construction. As usually, we initiate the model with Sequential():

#Initialising the CNN
classifier = Sequential()
"""
We specify our convolution layers and add MaxPooling to downsample and Dropout to prevent overfitting. We use Flatten and end with a Dense layer of 3 units, one for each class (circle [0], square [1], triangle [1]). We specify softmax as our last activation function, which is suggested for multiclass classification.
"""

#Step 1 - Convolution

classifier.add(Conv2D(32, (3, 3), padding='same', input_shape = (28, 28, 3), activation = 'relu'))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.5)) # antes era 0.25
​
# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.5)) # antes era 0.25
​
# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.5)) # antes era 0.25
​
# Step 3 - Flattening
classifier.add(Flatten())
​
# Step 4 - Full connection
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 3, activation = 'softmax'))

"""
For this type of images, I might be building an overly complex structure, and that will be evident once we take a look at the feature maps, however, for the sake of this article, it helps me to showcase exactly what each layer will be doing. I'm certain we can obtain the same or better results with less layers and less complexity.
"""

#Let's take a look at our model summary:

classifier.summary()
Compiling the CNN
classifier.compile(optimizer = 'rmsprop',
                   loss = 'categorical_crossentropy',
                   metrics = ['accuracy'])
"""
Using ImageDataGenerator to read images from directories
At this point we need to convert our pictures to a shape that the model will accept. For that we use the ImageDataGenerator. We initiate it and feed our images with .flow_from_directory. There are two main folders inside the working directory, called training_set and test_set. Each of those have 3 subfolders called circles, squares and triangles. I have sent 70 images of each shape to the training_set and 30 to the test_set.
"""

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
​
training_set = train_datagen.flow_from_directory('training_set',
                                                 target_size = (28, 28),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')
​
test_set = test_datagen.flow_from_directory('test_set',
                                            target_size = (28, 28),
                                            batch_size = 16,
                                            class_mode = 'categorical')
"""
Utilize callback to store the weights of the best model
The model will train for 30 epochs but we will use ModelCheckpoint to store the weights of the best performing epoch. We will specify val_acc as the metric to use to define the best model. This means we will keep the weights of the epoch that scores highest in terms of accuracy on the test set.
"""
checkpointer = ModelCheckpoint(filepath="best_weights.hdf5",
                               monitor = 'val_acc',
                               verbose=1,
                               save_best_only=True)

#Now it's time to train the model, here we include the callback to our checkpointer

history = classifier.fit_generator(training_set,
                                   steps_per_epoch = 100,
                                   epochs = 20,
                                   callbacks=[checkpointer],
                                   validation_data = test_set,
                                   validation_steps = 50)
"""
The model trained for 20 epochs but reached it's best performance at epoch 10. You will notice the message that says: Epoch 00010: val_acc improved from 0.93333 to 0.95556, saving model to best_weights.hdf5

That means we have now an hdf5 file which stores the weights of that specific epoch, where the accuracy over the test set was of 95,6%
"""
