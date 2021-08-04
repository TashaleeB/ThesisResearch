import numpy as np, matplotlib.pyplot as plt, datetime, tensorflow as tf, os
from tensorflow import keras
import tensorflow_probability as tfp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical, plot_model

toy_model = np.load('toy_models.npz')
nx, ny, ntrain = toy_model['training_data'].shape
training_data = toy_model['training_data'].T
labels = toy_model['labels']
print(training_data.shape)
print("nx", nx)
print("ny", ny)
print(labels[10])

example = training_data[10,:,:] #+
smex = spimg.gaussian_filter(example, 3)
noise = np.random.normal(0, 0.3, [32,32])
example_answer = labels[10]

# deterministic calculation of Tau of ideal image
print((example < 0.5).sum()/1024)
(training_data[10,:,:] == training_data[10,:,:].min()).sum()/1024
(training_data[10,:,:] < training_data[10,:,:].min()).sum()/1024

# Plot images
fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(15,6))
ax[0].imshow(example)
ax[1].imshow(smex)
ax[2].imshow(example-smex)
ax[3].imshow(example-smex+noise)

# Plot distribution
fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(15,4))
ax[0].hist(example.flatten(), bins=30);
ax[1].hist(smex.flatten(), bins=30);
ax[2].hist((example-smex).flatten(), bins=30);
ax[3].hist((example-smex+noise).flatten(), bins=30);

# Build model
model = Sequential()
# Add layers
model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(nx, ny, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(36, kernel_size=3, activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(200, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="relu"))
model.add(Dense(1,activation="linear"))

# Visualize Model
print(model.summary())
plot_model(model, show_shapes=True, show_layer_names=True)

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
model.compile(optimizer="adam", loss="mape")#, metrics=["accuracy"])
#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=100) #, callbacks=[tensorboard_callback])
predictions = model.predict(X_test).squeeze()

plt.plot(y_test, predictions, '.')
plt.plot(y_test,y_test, "r--")
plt.show()
