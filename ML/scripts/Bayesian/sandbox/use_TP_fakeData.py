# needs to be ran in hp_opt environment with Tensorflow version
# http://krasserm.github.io/2019/03/14/bayesian-neural-networks/
"""
List of Papers that use Bayesian CNN:

https://arxiv.org/pdf/1812.03973.pdf
https://arxiv.org/pdf/1806.05978.pdf
"""
# Data set: MNIST

# tf.__version__ : '2.1.0'

from __future__ import absolute_import, division, print_function

import os, warnings, matplotlib
# warnings.simplefilter(action="ignore")
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt, seaborn as sns, numpy as np, tensorflow as tf, tensorflow_probability as tfp

from matplotlib import figure
from matplotlib.backends import backend_agg
from tensorflow.keras.datasets import mnist
from keras import backend as K
from keras import activations, initializers
from keras.layers import Layer

import tensorflow as tf
import tensorflow_probability as tfp
tf.logging.set_verbosity(tf.logging.ERROR)

# Dependency imports
matplotlib.use("Agg")
%matplotlib inline


# Import Data MNIST data 3 different ways
# images reshaped to 784(28*28) vector, labels one-hot coded
mnist_onehot = input_data.read_data_sets(data_dir, one_hot=True)
# images not reshaped (28281 images), labels not one-hot coded
mnist_conv = input_data.read_data_sets(data_dir,reshape=False ,one_hot=False)
# images not reshaped, labels one-hot coded
mnist_conv_onehot = input_data.read_data_sets(data_dir,reshape=False ,one_hot=True)
# display an image
img_no = 485
one_image = mnist_conv_onehot.train.images[img_no].reshape(28,28)
plt.imshow(one_image, cmap='gist_gray')
print('Image label: {}'.format(np.argmax(mnist_conv_onehot.train.labels[img_no])))
