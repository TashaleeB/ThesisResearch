import tensorflow
import keras
from keras.datasets import mnist
# Use scikit-learn to grid search the batch size and epochs
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical

# download and split data into training and testing sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape and normalize the data
X_train = X_train.reshape(60000, 28, 28) / 255.0
X_test = X_test.reshape(10000, 28, 28) / 255.0

def create_model():
    # create model
    model = Sequential()

    # add layers
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    # compile our model
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# split into input (X) and output (Y) variables
X,Y = X_train, y_train #(60000, 28, 28), (60000,)

model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [2, 5, 10]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



"""
Using TensorFlow backend.
2019-11-07 10:01:07.188389: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2019-11-07 10:01:07.291777: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 0000:83:00.0
totalMemory: 11.17GiB freeMemory: 9.38GiB
2019-11-07 10:01:07.360376: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 1 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 0000:84:00.0
totalMemory: 11.17GiB freeMemory: 9.93GiB
2019-11-07 10:01:07.431680: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 2 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 0000:8a:00.0
totalMemory: 11.17GiB freeMemory: 9.82GiB
2019-11-07 10:01:07.504077: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 3 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 0000:8b:00.0
totalMemory: 11.17GiB freeMemory: 9.83GiB
2019-11-07 10:01:07.505023: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0, 1, 2, 3
2019-11-07 10:01:08.586013: I tensorflow/core/common_runtime/gpu/gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-11-07 10:01:08.586093: I tensorflow/core/common_runtime/gpu/gpu_device.cc:929]      0 1 2 3
2019-11-07 10:01:08.586105: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 0:   N Y Y Y
2019-11-07 10:01:08.586111: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 1:   Y N Y Y
2019-11-07 10:01:08.586117: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 2:   Y Y N Y
2019-11-07 10:01:08.586123: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 3:   Y Y Y N
2019-11-07 10:01:08.586607: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 9090 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:83:00.0, compute capability: 3.7)
2019-11-07 10:01:08.721720: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 9625 MB memory) -> physical GPU (device: 1, name: Tesla K80, pci bus id: 0000:84:00.0, compute capability: 3.7)
2019-11-07 10:01:08.854083: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:2 with 9518 MB memory) -> physical GPU (device: 2, name: Tesla K80, pci bus id: 0000:8a:00.0, compute capability: 3.7)
2019-11-07 10:01:08.991700: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:3 with 9524 MB memory) -> physical GPU (device: 3, name: Tesla K80, pci bus id: 0000:8b:00.0, compute capability: 3.7)

In [2]: QStandardPaths: XDG_RUNTIME_DIR points to non-existing path '/run/user/71578', please create it with 0700 permissions.
"""
