"""
Beginner TF: https://www.datacamp.com/community/tutorials/cnn-tensorflow-python

https://matthewmcateer.me/blog/a-quick-intro-to-bayesian-neural-networks/


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

# Dependency imports
import numpy as np, matplotlib
import tensorflow as tf
import tensorflow_probability as tfp

matplotlib.use("Agg")
from matplotlib import figure
from matplotlib.backends import backend_agg
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

warnings.simplefilter(action="ignore")

tfd = tfp.distributions

IMAGE_SHAPE = [28, 28, 1]

learning_rate = 0.001   #initial learning rate
max_step = 5000 #number of training steps to run
batch_size = 50 #batch size
viz_steps = 500 #frequency at which save visualizations.
num_monte_carlo = 50 #Network draws to compute predictive probabilities.
  

# download and split data into training and testing sets
#train, test = mnist.load_data() # not reshaped or one-hot encoded
(X_train, y_train), (X_test, y_test) = mnist.load_data() # not reshaped or one-hot encoded

# reshape and normalize the data
X_train_norm = X_train.reshape(60000, 28, 28) / 255.0
X_test_norm = X_test.reshape(10000, 28, 28) / 255.0

# one-hot encode target column
y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)

# construct a Dataset from data read into memory, you can use from_tensors()
X_train_norm = tf.data.Dataset.from_tensors(X_train_norm)
y_train_onehot = tf.data.Dataset.from_tensors(y_train_onehot)
X_test_norm = tf.data.Dataset.from_tensors(X_test_norm)
y_test_onehot = tf.data.Dataset.from_tensors(y_test_onehot)

# construct a Dataset from data read into memory, you can use from_tensors()
X_train = tf.data.Dataset.from_tensors(X_train)
y_train = tf.data.Dataset.from_tensors(y_train)
X_test = tf.data.Dataset.from_tensors(X_test)
y_test = tf.data.Dataset.from_tensors(y_test)


images = tf.placeholder(tf.float32,shape=[None,28,28,1])
labels = tf.placeholder(tf.float32,shape=[None,])
hold_prob = tf.placeholder(tf.float32)


# define the model
neural_net = tf.keras.Sequential([
      tfp.layers.Convolution2DReparameterization(32, kernel_size=5,  padding="SAME", activation=tf.nn.relu),
      tf.keras.layers.MaxPooling2D(pool_size=[2, 2],  strides=[2, 2],  padding="SAME"),
      tfp.layers.Convolution2DReparameterization(64, kernel_size=5,  padding="SAME",  activation=tf.nn.relu),
      tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME"),
      tf.keras.layers.Flatten(),
      tfp.layers.DenseFlipout(1024, activation=tf.nn.relu),
      tf.keras.layers.Dropout(hold_prob),
      tfp.layers.DenseFlipout(10)])

# Parameterize by logits rather than probabilities.
# logit or log-odds is the logarithm of the odds, p/(1-p), where p is a probability
logits = neural_net(images)

# Compute the -ELBO as the loss, averaged over the batch size.
labels_distribution = tfp.distributions.Categorical(logits=logits) # generalized Bernoulli distribution (multinoulli distribution) for 10 possible outcomes
neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(labels)) # negative mean of the Log probability mass function of float
kl = sum(neural_net.losses) / len(y_train)#mnist_conv.train.num_examples
elbo_loss = neg_log_likelihood + kl
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss = elbo_loss) # An Operation that updates the variables in var_list

# Build metrics for evaluation. Predictions are formed from a single forward
# pass of the probabilistic layers. They are cheap but noisy predictions.
predictions = tf.argmax(logits, axis=1)

# A Tensor representing the accuracy, the value of total divided by count.
# An operation that increments the total and count variables appropriately and whose value matches accuracy.
accuracy, accuracy_update_op = tf.metrics.accuracy(labels=labels, predictions=predictions)

# creates op that groups multiple operations
init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())

with tf.Session() as sess:
        sess.run(init_op) # Runs operations and evaluates tensors in fetches
        
# Run the training loop.
        for step in range(max_step+1):
            images_b, labels_b = mnist_conv.train.next_batch(batch_size)
            images_h, labels_h = mnist_conv.validation.next_batch(mnist_conv.validation.num_examples)
            
            _ = sess.run([train_op, accuracy_update_op], feed_dict={images: images_b,labels: labels_b,hold_prob:0.5})
            
        if (step==0) | (step % 500 == 0):
                loss_value, accuracy_value = sess.run([elbo_loss, accuracy],
                feed_dict={images: images_b, labels: labels_b,hold_prob:0.5})
                
                print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(step, loss_value, accuracy_value))
















# download and split data into training and testing sets
(conv_X_train, conv_y_train), (conv_X_test, conv_y_test) = mnist.load_data()

# No reshaping with one-hot encode target column
conv_onehot_X_train = conv_X_train
conv_onehot_y_train = to_categorical(conv_y_train)
conv_onehot_X_test = conv_X_test
conv_onehot_y_test = to_categorical(conv_y_test)

# reshape and normalize the data
X_train = conv_X_train.reshape(60000, 28, 28) / 255.0
X_test = conv_X_test.reshape(10000, 28, 28) / 255.0

# one-hot encode target column
onehot_y_train = to_categorical(conv_y_train)
onehot_y_test = to_categorical(conv_y_test)

list(tf.data.Dataset.from_tensor_slices(conv_X_train).batch(64, drop_remainder=False).as_numpy_iterator())
