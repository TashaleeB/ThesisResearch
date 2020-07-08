# needs to be ran in hp_opt environment with Tensorflow version
# https://towardsdatascience.com/bayesian-neural-networks-with-tensorflow-probability-fbce27d6ef6
# Data set: http://archive.ics.uci.edu/ml/datasets/Air+Quality

# tf.__version__ : '2.1.0'

from __future__ import print_function, division, absolute_import

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
tf.keras.backend.set_floatx("float64")
tfd = tfp.distributions

from sklearn.preprocessing import StandardScaler

# An algorithm:
# randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature
from sklearn.ensemble import IsolationForest

import gc
gc.enable()
