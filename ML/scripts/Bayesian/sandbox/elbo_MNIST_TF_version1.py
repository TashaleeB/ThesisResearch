"""
Beginner TF: https://www.datacamp.com/community/tutorials/cnn-tensorflow-python

https://medium.com/python-experiments/bayesian-cnn-model-on-mnist-data-using-tensorflow-probability-compared-to-cnn-82d56a298f45


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
from tensorflow.python.framework.ops import disable_eager_execution
warnings.simplefilter(action="ignore")

tfd = tfp.distributions

IMAGE_SHAPE = [28, 28, 1]

learning_rate = 0.001 # Initial learning rate.
max_steps = 6000 #Number of training steps to run.
batch_size = 128 #Batch size.
# Directory to put the model's fit.
model_dir = "/ocean/projects/ast180004p/tbilling/sandbox/bayesian/denseflipout/sandbox/"
viz_steps = 400 #Frequency at which save visualizations.
num_monte_carlo = 50 #Network draws to compute predictive probabilities.
fake_data = True #If true, uses fake data. Defaults to real data.


def plot_weight_posteriors(names, qm_vals, qs_vals, fname):
    fig = figure.Figure(figsize=(6, 3))
    canvas = backend_agg.FigureCanvasAgg(fig)

    ax = fig.add_subplot(1, 2, 1)
    for n, qm in zip(names, qm_vals):
        sns.distplot(qm.flatten(), ax=ax, label=n)
    ax.set_title("weight means")
    ax.set_xlim([-1.5, 1.5])
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    for n, qs in zip(names, qs_vals):
        sns.distplot(qs.flatten(), ax=ax)
    ax.set_title("weight stddevs")
    ax.set_xlim([0, 1.])

    fig.tight_layout()
    canvas.print_figure(fname, format="png")
    print("saved {}".format(fname))


def plot_heldout_prediction(input_vals, probs, fname, n=10, title=""):
    fig = figure.Figure(figsize=(9, 3*n))
    canvas = backend_agg.FigureCanvasAgg(fig)
    for i in range(n):
        ax = fig.add_subplot(n, 3, 3*i + 1)
        ax.imshow(input_vals[i, :].reshape(IMAGE_SHAPE[:-1]), interpolation="None")

        ax = fig.add_subplot(n, 3, 3*i + 2)
        for prob_sample in probs:
            sns.barplot(np.arange(10), prob_sample[i, :], alpha=0.1, ax=ax)
            ax.set_ylim([0, 1])
        ax.set_title("posterior samples")

        ax = fig.add_subplot(n, 3, 3*i + 3)
        sns.barplot(np.arange(10), np.mean(probs[:, i, :], axis=0), ax=ax)
        ax.set_ylim([0, 1])
        ax.set_title("predictive probs")
    fig.suptitle(title)
    fig.tight_layout()

    canvas.print_figure(fname, format="png")
    print("saved {}".format(fname))

# Function that builds fake data
def build_fake_data(num_examples=10):
    class Dummy(object):
        pass

    num_examples = 10
    mnist_data = Dummy()
    mnist_data.train = Dummy()
    mnist_data.train.images = np.float32(np.random.randn(num_examples, *IMAGE_SHAPE))
    mnist_data.train.labels = np.int32(np.random.permutation(np.arange(num_examples)))
    mnist_data.train.num_examples = num_examples
    mnist_data.validation = Dummy()
    mnist_data.validation.images = np.float32(np.random.randn(num_examples, *IMAGE_SHAPE))
    mnist_data.validation.labels = np.int32(np.random.permutation(np.arange(num_examples)))
    mnist_data.validation.num_examples = num_examples
    return mnist_data


mnist_data = build_fake_data()


def build_input_pipeline(mnist_data, batch_size, heldout_size):
    training_dataset = tf.data.Dataset.from_tensor_slices((mnist_data.train.images, np.int32(mnist_data.train.labels)))
    training_batches = training_dataset.shuffle(50000, reshuffle_each_iteration=True).repeat().batch(batch_size)
    training_iterator = tf.compat.v1.data.make_one_shot_iterator(training_batches)

    heldout_dataset = tf.data.Dataset.from_tensor_slices((mnist_data.validation.images, np.int32(mnist_data.validation.labels)))
    heldout_frozen = (heldout_dataset.take(heldout_size).repeat().batch(heldout_size))
    heldout_iterator = tf.compat.v1.data.make_one_shot_iterator(heldout_frozen)

    # https://stackoverflow.com/questions/53429896/how-do-i-disable-tensorflows-eager-execution
    #tf.compat.v1.disable_eager_execution()
    disable_eager_execution()
    handle = tf.compat.v1.placeholder(tf.string, shape=[])
    feedable_iterator = tf.compat.v1.data.Iterator.from_string_handle(handle, (tf.float32, tf.int32), ((None, 28, 28, 1), (None,)))
    images, labels = feedable_iterator.get_next()

    return images, labels, handle, training_iterator, heldout_iterator


(images, labels, handle, training_iterator, heldout_iterator) = build_input_pipeline(mnist_data = mnist_data, batch_size = batch_size, heldout_size = mnist_data.validation.num_examples)

with tf.compat.v1.name_scope("bayesian_neural_net", values=[images]):
    neural_net = tf.keras.Sequential([
        tfp.layers.Convolution2DFlipout(6,
                                        kernel_size=5,
                                        padding="SAME",
                                        activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                     strides=[2, 2],
                                     padding="SAME"),
        tfp.layers.Convolution2DFlipout(16,
                                        kernel_size=5,
                                        padding="SAME",
                                        activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                     strides=[2, 2],
                                     padding="SAME"),
        tfp.layers.Convolution2DFlipout(120,
                                        kernel_size=5,
                                        padding="SAME",
                                        activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        tfp.layers.DenseFlipout(84, activation=tf.nn.relu),
        tfp.layers.DenseFlipout(10)
        ])
      
    logits = neural_net(images)
    labels_distribution = tfd.Categorical(logits=logits)

# Compute the -ELBO as the loss, averaged over the batch size.
neg_log_likelihood = -tf.reduce_mean(input_tensor=labels_distribution.log_prob(labels))
kl = sum(neural_net.losses) / mnist_data.train.num_examples
elbo_loss = neg_log_likelihood + kl

# Predictions are formed from a single forward pass of the probabilistic layers.
predictions = tf.argmax(input=logits, axis=1)
accuracy, accuracy_update_op = tf.compat.v1.metrics.accuracy(labels=labels, predictions=predictions)

# Exct weight posterior statistics for layers with weight distributions for later visualization
names = []
qmeans = []
qstds = []
for i, layer in  enumerate(neural_net.layers):
    try:
        q = layer.kernel_posterior
    except  AttributeError:
        continue
    names.append("Layer {}".format(i))
    qmeans.append(q.mean())
    qstds.append(q.stddev())

# training loop
with tf.compat.v1.name_scope("train"):
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(elbo_loss)
  
init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
  
with tf.compat.v1.Session() as sess:
    sess.run(init_op)
      
    # Run the training loop.
    train_handle = sess.run(training_iterator.string_handle())
    heldout_handle = sess.run(heldout_iterator.string_handle())
    for step in  range(max_steps):
        _ = sess.run([train_op, accuracy_update_op],
                     feed_dict={handle: train_handle})
          
        if step % 100 == 0:
            loss_value, accuracy_value = sess.run(
                [elbo_loss, accuracy], feed_dict={handle: train_handle})
            print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(step, loss_value, accuracy_value))
          
        if (step+1) % viz_steps == 0:
            # Compute log prob of heldout set by averaging draws from the model:
            # p(heldout | train) = int_model p(heldout|model) p(model|train)
            # ~= 1/n * sum_{i=1}^n p(heldout | model_i)
            # where model_i is a draw from the posterior p(model|train).
            probs = np.asarray([sess.run((labels_distribution.probs),
                                feed_dict={handle: heldout_handle}) for _ in  range(num_monte_carlo)])
            mean_probs = np.mean(probs, axis=0)
              
            image_vals, label_vals = sess.run((images, labels),
                                              feed_dict={handle: heldout_handle})
            heldout_lp = np.mean(np.log(mean_probs[np.arange(mean_probs.shape[0]),
                                    label_vals.flatten()]))
            print(" ... Held-out nats: {:.3f}".format(heldout_lp))
              
            qm_vals, qs_vals = sess.run((qmeans, qstds))
              
            plot_weight_posteriors(names, qm_vals, qs_vals,
                                   fname=os.path.join(
                                    model_dir,
                                    "step{:05d}_weights.png"
                                    .format(step)))

            plot_heldout_prediction(image_vals, probs,
                                    fname=os.path.join(
                                        model_dir,
                                        "step{:05d}_pred.png".format(step)),
                                        title="mean heldout logprob {:.2f}"
                                        .format(heldout_lp))
