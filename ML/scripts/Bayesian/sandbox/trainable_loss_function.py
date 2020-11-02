# -*- coding: utf-8 -*-
# https://stackoverflow.com/questions/58566096/custom-loss-function-that-updates-at-each-step-via-gradient-descent

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Input, Layer, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal, RandomUniform
from tensorflow.keras.datasets import boston_housing

# define a layer for storing the values that go into our loss parameters
class TrainableLossLayer(Layer):
    """
    A class for storing parameters that go into our loss function.

    This class is needed for storing values that are updated during training. We
    define several "required" methods on it so that it can be inserted into a
    keras model properly.

    Parameters
    ----------
    initializer : keras initializer object
        An initializer for generating the initial values of the parameters.

    Returns
    -------
    Layer
        A keras Layer that is suitable for input to a keras graph.
    """
    def __init__(self, initializer, **kwargs):
        super(TrainableLossLayer, self).__init__(**kwargs)
        self.initializer = initializer

    def build(self, input_shape):
        """
        Define what goes into this layer.

        Parameters
        ----------
        input_shape : tuple
            The input shape that should be used for this layer. This is a
            required parameter for a keras layer, even though we're not going to
            explicitly use it in our case.

        Returns
        -------
        None
        """
        self.kernel = self.add_weight(
            name="kernel", shape=(1,), initializer=self.initializer, trainable=True
        )
        self.built = True

    def call(self, inputs):
        """
        Define what this layer should do when it's in a graph.

        In our case, this will just return the value that's currently stored in
        the weight.

        Parameters
        ----------
        inputs : keras layer
            The input that this layer takes. Again, this is a required parameter
            for a custom keras layer that we won't explicitly use.

        Returns
        -------
        keras tensor
            The value that this layer provides in the computational graph.
        """
        return self.kernel

    def compute_output_shape(self, input_shape):
        """
        Define what the output shape of this layer is.

        Because this only puts out a single value (the current value of the
        weight), it's always (1,).

        Parameters
        ----------
        input_shape : tuple
            The input shape. Ignored dummy parameter.

        Returns
        -------
        tuple
            The shape of the output from this layer.
        """
        return (1,)

    def get_config(self):
        """
        Define what the configuration is for this layer.

        When we go to save the model, we need to communicate to keras that there
        is a required parameter that the __init__ function takes. Otherwise it
        will complain and not save our model.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        config = super().get_config().copy()
        config.update(
            {"initializer": self.initializer}
        )

# okay, let's get the inputs we'll use for the model
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

# Now we define Input layers that will hold these data.
# We're going to do something funky and pass the y data in as *input data*,
# because we're going to pass over the normal way of defining our loss
# function. Just bear with me for now...
x_input = Input(X_train.shape[1:2])
y_input = Input(shape=(1,))

# Now, we need to define initializers for our mu and sigma values.
# Mu can be positive or negative, so we use a default "normal" value.
# Sigma must be positive, so we start it out there.
# In practice, we may need to update these to get better results...
mu_initializer = RandomNormal(mean=0.0, stddev=1.0)
sigma_initializer = RandomUniform(minval=1.0, maxval=10.0)

# Now, we actually *make* the mu and sigma layers we'll need.
# The input layer we pass in doesn't matter, because it'll be ignored.
mu = TrainableLossLayer(mu_initializer, name="mu_value")(x_input)
sigma = TrainableLossLayer(sigma_initializer, name="sigma_value")(x_input)

# Now we define the "likelihood" loss function. Recall that we should use
# keras backend functions instead of normal numpy operations.
def lambda_loss(x):
    """
    Define our loss function.

    This function will be called as a lambda layer, so that we can pass in
    multiple values to it.

    Parameters
    ----------
    x : tuple of input data
        Our input arguments, which will be unpacked.

    Returns
    -------
    keras tensor
        The value of the loss function.
    """
    y_true, y_pred, mu, sigma = x
    likelihood = 1.0 / K.sqrt(
        2 * np.pi * (sigma * y_true)**2
    ) * K.exp(
        -(y_true - y_pred - mu)**2 / (sigma * y_true)**2
    )
    # take negative log of likelihood, since we're *minimizing* the loss
    return -K.log(likelihood)

# Okay, finally ready to make our model!
# This is a simple model with 2 hidden dense layers, no fancy convolutions.
# Also note that I'm using the "Functional API" of Keras, where I use Model()
# instead of Sequential(). This is necessary for the loss layers to behave
# properly.
d1 = Dense(8)(x_input)
d2 = Dense(16)(d1)
d3 = Dense(1)(d2)

# The "output" value of the network is in the "d3" layer. So that's the "y_pred"
# that I'm going to use in my loss function. So I'm going to call a Lambda layer
# with the right arguments now, which will be my loss function.
loss = Lambda(lambda_loss)([y_input, d3, mu, sigma])

# Now I define my model. The input is my X values *and* my snapshot labels. The
# output is actually the result of my loss function.
model = Model([x_input, y_input], loss)

# Now I need to define a dummy loss function for my optimizer. Again, because
# the output of my network is my *actual* loss function, I don't need to do
# anything here.
def dummy_loss(y_true, y_pred):
    """
    Define a dummy loss function.

    As with all keras losses, I have the required arguments of y_true and
    y_pred.

    Parameters
    ----------
    y_true : keras tensor
        The "true" values of my input data.
    y_pred : keras tensor
        The "predicted" values of my network.

    Returns
    -------
    keras tensor
        The "loss" values.
    """
    return y_pred

# Now to compile our model...
model.compile(optimizer="adam", loss=dummy_loss)

# Let's see what it looks like
model.summary()

# Finally, our model needs fake input data to do back propagation, and isn't
# smart enough to know that it doesn't actually need it. So we just make zeros
# to fake it out.
dummy_train = np.zeros_like(y_train)
dummy_test = np.zeros_like(y_test)

# Now we can finally train!
model.fit(
    [X_train, y_train],
    dummy_train,
    validation_data=([X_test, y_test], dummy_test),
    epochs=100,
)

# Let's see what the values of mu and sigma are.
# The actualy location of the layers might change, you should check the output
# of model.summary()
mu_layer = model.layers[-3]
mu_value = mu_layer.get_weights()
print("mu value: ", mu_value)
sigma_layer = model.layers[-2]
sigma_value = sigma_layer.get_weights()
print("sigma value: ", sigma_value)

# Finally, let's save our model.
model.save("my_model.h5")
