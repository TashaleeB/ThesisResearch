# needs to be ran in hp_opt environment with Tensorflow version
# http://krasserm.github.io/2019/03/14/bayesian-neural-networks/

# tf.__version__ : '2.1.0'

import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K
from keras import activations, initializers, callbacks, optimizers
from keras.layers import Layer, Input
from keras.models import Model
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow_probability as tfp

np.random.seed(12252020)
tf.random.set_seed(12252020)

train_size = 32
noise = 1.0
X = np.linspace(-0.5, 0.5, train_size).reshape(-1, 1)
X_test = np.linspace(-1.5, 1.5, 1000).reshape(-1, 1) # (1000, 1)

"""
The training dataset consists of 32 noisy samples X, y drawn from a sinusoidal function.
"""
def f(x, sigma):
    epsilon = np.random.randn(*x.shape) * sigma
    return 10 * np.sin(2 * np.pi * (x)) + epsilon

y = f(X, sigma=noise)
y_true = f(X, sigma=0.0)

"""
Make a plot of the true data (some continuous function) and the training data which is reepresented by some marker. The training data has some random noise
"""
plt.figure()
plt.scatter(X, y, marker='+', label='Training data')
plt.plot(X, y_true, label='Truth')
plt.title('Noisy training data and ground truth')
plt.legend()
plt.show()

class DenseVariational(Layer):

    """
    The noise in training data gives rise to aleatoric uncertainty. To cover epistemic uncertainty the variational inference logic is implemented in a custom DenseVariational Keras layer. The complexity cost (kl_loss) is computed layer-wise and added to the total loss with the add_loss method. Implementations of build and call directly follow the equations defined above.
    """

    def __init__(self,
                 units,
                 kl_weight,
                 activation=None,
                 prior_sigma_1=1.5,
                 prior_sigma_2=0.1,
                 prior_pi=0.5, **kwargs):
        self.units = units
        self.kl_weight = kl_weight
        self.activation = activations.get(activation)
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi_1 = prior_pi
        self.prior_pi_2 = 1.0 - prior_pi
        self.init_sigma = np.sqrt(self.prior_pi_1 * self.prior_sigma_1 ** 2 +
                                  self.prior_pi_2 * self.prior_sigma_2 ** 2)

        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def build(self, input_shape):
        self.kernel_mu = self.add_weight(name='kernel_mu',
                                         shape=(input_shape[1], self.units),
                                         initializer=initializers.RandomNormal(stddev=self.init_sigma),
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu',
                                       shape=(self.units,),
                                       initializer=initializers.RandomNormal(stddev=self.init_sigma),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho',
                                          shape=(input_shape[1], self.units),
                                          initializer=initializers.constant(0.0),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho',
                                        shape=(self.units,),
                                        initializer=initializers.constant(0.0),
                                        trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)

        bias_sigma = tf.math.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)

        self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) +
                      self.kl_loss(bias, self.bias_mu, bias_sigma))

        return self.activation(K.dot(inputs, kernel) + bias)

    def kl_loss(self, w, mu, sigma):
        variational_dist = tfp.distributions.Normal(mu, sigma)
        return self.kl_weight * K.sum(variational_dist.log_prob(w) - self.log_prior_prob(w))

    def log_prior_prob(self, w):
        comp_1_dist = tfp.distributions.Normal(0.0, self.prior_sigma_1)
        comp_2_dist = tfp.distributions.Normal(0.0, self.prior_sigma_2)
        return K.log(self.prior_pi_1 * comp_1_dist.prob(w) +
                     self.prior_pi_2 * comp_2_dist.prob(w))

"""
Since the training dataset has only train_size examples we train the network with all train_size examples per epoch so that the number of batches per epoch is 1. For other configurations, the complexity cost (kl_loss) must be weighted by 1/M where M is the number of mini-batches per epoch. The hyper-parameter values for the mixture prior, prior_params, have been chosen to work well for this example and may need adjustments in another context.
"""

batch_size = train_size
num_batches = train_size / batch_size

kl_weight = 1.0 / num_batches
prior_params = {
    'prior_sigma_1': 1.5,
    'prior_sigma_2': 0.1,
    'prior_pi': 0.5}

x_in = Input(shape=(1,))
x = DenseVariational(20, kl_weight, **prior_params, activation='relu')(x_in)
x = DenseVariational(20, kl_weight, **prior_params, activation='relu')(x)
x = DenseVariational(1, kl_weight, **prior_params)(x)

model = Model(x_in, x)

"""
The network can now be trained with a Gaussian negative log likelihood function (neg_log_likelihood) as loss function assuming a fixed standard deviation (noise).
"""
def neg_log_likelihood(y_obs, y_pred, sigma=noise):
    # Define a batch of some number of scalar valued Normals.
    dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
    # Evaluate the pdf of both distributions on the same point, 3.0
    return K.sum(-dist.log_prob(y_obs))

model.compile(loss=neg_log_likelihood, optimizer=optimizers.Adam(lr=0.08), metrics=['mse'])
model.fit(X, y, batch_size=batch_size, epochs=1500, verbose=2)

"""
When calling model.predict we draw a random sample from the variational posterior distribution and use it to compute the output value of the network. This is equivalent to obtaining the output from a single member of a hypothetical ensemble of neural networks. Drawing 500 samples means that we get predictions from 500 ensemble members. From these predictions we can compute statistics such as the mean and standard deviation. In our example, the standard deviation is a measure of epistemic uncertainty.
"""
y_pred_list = [model.predict(X_test) for i in np.arange(500)] # (500, 1000, 1)
y_preds = np.concatenate(y_pred_list, axis=1) # (1000, 500)

y_mean = np.mean(y_preds, axis=1) # (1000, 1)
y_sigma = np.std(y_preds, axis=1) # (1000, 1)

plt.plot(X_test, y_mean, 'r-', label='Predictive mean');
plt.scatter(X, y, marker='+', label='Training data')
plt.fill_between(X_test.ravel(),
                 y_mean + 2 * y_sigma,
                 y_mean - 2 * y_sigma,
                 alpha=0.5, label='Epistemic Uncertainty')
plt.title('Prediction')
plt.legend()
plt.show()

"""
-------------------------------------------------------------------------------------------
The section below allows you to calculate both Epistemic and Aleatoric Uncertainties.
-------------------------------------------------------------------------------------------
"""

negloglik = lambda y, rv_y: -rv_y.log_prob(y)

batch_size = train_size
num_batches = train_size / batch_size

kl_weight = 1.0 / num_batches

# Build model.
model = tf.keras.Sequential([
  tfp.layers.DenseVariational(20, posterior_mean_field, prior_trainable, kl_weight=kl_weight),
  tfp.layers.DenseVariational(20, posterior_mean_field, prior_trainable, kl_weight=kl_weight),
  tfp.layers.DenseVariational(1+1, posterior_mean_field, prior_trainable, kl_weight=kl_weight),
  tfp.layers.DistributionLambda(
      lambda t: tfd.Normal(loc=t[..., :1],
                           scale=1e-3 + tf.math.softplus(0.01 * t[...,1:]))),
])

# Do inference.
model.compile(loss=negloglik, optimizer=optimizers.Adam(lr=0.08), metrics=['mse'])
model.fit(X, y, batch_size=batch_size, epochs=1500, verbose=2)
model(X_test[0])
