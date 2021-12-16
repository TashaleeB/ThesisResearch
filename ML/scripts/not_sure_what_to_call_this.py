# Import Libraries
import numpy as np, matplotlib.pyplot as plt, seaborn as sns, keras_tuner as kt, tensorflow as tf
import os, datetime, progressbar, copy
import tensorflow_probability as tfp

tfd = tfp.distributions

from progressbar import ProgressBar
pbar = ProgressBar()

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical, plot_model
from keras.callbacks import ModelCheckpoint#, TensorBoard
from keras import backend as K
from livelossplot import PlotLossesKeras

from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hypermodel import HyperModel
from keras_tuner.engine.hyperparameters import HyperParameters
#from tensorboard.plugins.hparams import api as hp
#from keras.utils.generic_utils import get_custom_objects

import gc
gc.enable()

import warnings
warnings.filterwarnings("ignore")

hp = HyperParameters()
plotlosses = PlotLossesKeras()

var_nomode = [.2, .02, .002]
var_mode = [0.01, 0.001, 1e-4 ]

# Normalization term for divergence
def kl_divergence_normalization_term(TRAINING_DATA):
    """
    Parameters
    ----------
    TRAINING_DATA : ndarray
        Input image data. Will be converted to int
    """

    NUM_TRAIN_EXAMPLES = len(TRAINING_DATA)
    kl_divergence_function = (
    lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(NUM_TRAIN_EXAMPLES,dtype=tf.float32)
    )
    return kl_divergence_function



"""
Parameters
----------
image : ndarray
    Input image data. Will be converted to float.
mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
"""

class custom_loss:

    """
    Parameters
    ----------
    y_true : ndarray
        Input array dataset. Will be converted to float.
    y_pred:
        Input array dataset. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
    """

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    # Custom Loss Functions
    kl_divergence = tf.keras.losses.KLDivergence()

    neg_log_likelihood = lambda y_true, y_pred: -tf.reduce_mean(input_tensor=y_pred.log_prob(y_true))

    def Mean_Squared_over_true_Error(y_true, y_pred):
        # Create a custom loss function that divides the difference by the true

        y_true = K.cast(y_true, y_pred.dtype) #Casts a tensor to a different dtype and returns it.
        diff_ratio = K.square((y_pred - y_true)/K.clip(K.abs(y_true),K.epsilon(),None))

        loss = K.mean(diff_ratio, axis=-1)

        # Return a tensor
        return loss

    def elbo(y_true, y_pred):
        kl_weight = 1
        neg_log_likelihood = -tf.reduce_mean(input_tensor=y_pred.log_prob(y_true))
        kl_divergence = tf.keras.losses.KLDivergence()

        elbo_loss = -tf.math.reduce_mean(-kl_weight * kl_divergence(y_true, y_pred.mean()) - neg_log_likelihood)
        # Return a tensor
        return elbo_loss

    def mean_fractional_error(y_true, y_pred):
        y_true = K.cast(y_true, y_pred.dtype) #Casts a tensor to a different dtype and returns it.
        diff_ratio = (y_pred - y_true)/K.clip(K.abs(y_true),K.epsilon(),None)
        loss = K.mean(diff_ratio, axis=-1)
        
        return loss

    keras.utils.get_custom_objects().update({'Mean_Squared_over_true_Error':Mean_Squared_over_true_Error,
                                            'neg_log_likelihood': neg_log_likelihood, 'kl_divergence':kl_divergence,
                                            'elbo':elbo})
                                            

def model1_ploss(X_train, y_train, X_test, y_test, nx, ny, kl_divergence_function, epochs, batch_size):
    # Making a model.
    model = Sequential()
    
    model.add(tfp.layers.Convolution2DFlipout(filters = 16,
                                              input_shape=(nx, ny, 1), kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              #, padding = 'same'
                                             ))
    model.add(tfp.layers.Convolution2DFlipout(filters = 8,
                                              kernel_size=3, activation = 'relu',
                                              kernel_divergence_fn=kl_divergence_function,
                                              #padding = 'same'
                                             ))
    model.add(Flatten())
    model.add(tfp.layers.DenseFlipout(1, activation="linear", kernel_divergence_fn=kl_divergence_function,))
    model.add(tfp.layers.DistributionLambda(
      lambda t: tfd.Normal(loc=t[..., :1],
                           #scale = 1)))
                           scale=1e-3 + tf.math.softplus(0.01 * t[..., :1]))))
    

    model.compile(optimizer= "adam", loss = "neg_log_likelihood", metrics=["mse"])
    
    plotlosses = PlotLossesKeras()

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
            epochs=epochs, batch_size=batch_size,
            callbacks=[plotlosses],
            verbose=1)
    
    # Visualize Model
    print(model.summary())

    loss = model.evaluate(X_test, y_test)
    return model

def random_search(model,num_models,outputdir,project_name, X_train, y_train, X_test, y_test, epochs,HP_BATCH_SIZE):
    # Start Tuning based on low validation loss
    tuner = RandomSearch(model,
                         objective='val_loss', # 'loss', 'val_loss', 'val_accuracy'
                         max_trials= num_models,# specify the number of different models to try
                         executions_per_trial=1, #2,
                         directory=outputdir,
                         seed=42,
                         project_name=project_name)

    # Print a summary of the search space
    tuner.search_space_summary()

    # Show the best models, their hyperparameters, and the resulting metrics.
    tuner.search(X_train, y_train,
                 epochs=epochs,#hp.Int('epoch',min_value=10,max_value=HP_EPOCH,step=10,default=10),
                 validation_data=(X_test, y_test),
                 batch_size=HP_BATCH_SIZE,
                #callbacks=[plotlosses],
                verbose=1)
                
    print("-"*50)
    print("Tuning Summary")
    print("-"*50)
    tuner.results_summary()
    tuner.search_space_summary()

    # Retrieve the best model.
    #best_model = tuner.get_best_models(num_models=1)[0]

    return tuner.get_best_models(num_models=20)
 
"""
def save_tuned_models(random_search_models=tuner.get_best_models(num_models=num_models), outputdir, num_models, num_of_models= len(tuner.get_best_models(num_models=20))):
    
    # Save best hyperperam Model
    print("saving best hyperparameter model ...")
    best_model.save(outputdir+"hyperParam_model_256.h5")

    # Save best hyperperam Model Weights
    print("saving best hyperparameter model weights ...")
    best_model.save_weights(outputdir+"hyperParam_model_weights_256.h5")


    # Save all other models so that we can see how the model behaves based on complexity
    for indx in np.arange(num_models, num_of_models):
        model_ = random_search_models[indx]
        # Save Model weights
        model_.save_weights(outputdir+"hyperParam_weights_{}_{}.h5".format(str(indx+1),idx))
        print("Saving ... "+outputdir+"hyperParam_weights_{}_{}.h5".format(str(indx+1),idx))
        # Save Model
        model_.save(outputdir+"hyperParam_model_{}_{}.h5".format(str(indx+1),idx))
        print("Saving ... "+outputdir+"hyperParam_model_{}_{}.h5".format(str(indx+1),idx))
"""


def mc_pred(bestmodel, X_test):
    #bestmodel=tuner.get_best_models(num_models=1)[0]#.predict(X_test)
    pred = []

    for i in range(500):#pbar(range(100)):
        y_p = bestmodel.predict(X_test).squeeze()#predict(X_test, batch_size=test_labels.shape[0])
        pred.append(y_p)

    return pred

def mc_pred_tuner_models(tuner_models, X_test):
    # make predictions
    prediction = []
    predictions = np.empty([len(X_test),])

    for m in tuner_models:
        for i in range(500):#pbar(range(500)):
            y_p = m.predict(X_test).squeeze()#predict(X_test, batch_size=test_labels.shape[0])
            prediction.append(y_p) # (500, 100, 1) = (# of masks, # of datasets, # of classes)
        prediction = np.mean(np.array(prediction), axis=0)
        predictions = np.vstack((predictions, prediction))
        prediction = []
    print(X_test.shape)
    print(predictions.shape)

    
    return predictions


def plot_one_to_one_pred(tuner_models, y_test, predictions):
    # Plot one-to-one predictions
    plt.figure(figsize=(15,15))
    plt.title("Compare Predictions", size = 20)

    for l in range(len(tuner_models)):
        plt.plot(y_test, predictions[l], '.', label = str(l+1))
    plt.plot(y_test,y_test, "k--", linewidth=5)

    plt.xlabel("Ionized Pixels", size = 16)
    plt.ylabel("Predictions", size = 16)
    plt.legend(markerscale=2.5)
    plt.tight_layout()

def plot_pred_distribution(predictions, y_test):
    # data from mc_pred
 
    plt.figure(figsize=(15, 10))
    plt.title("Predicted Number of Ionized {}".format("Pixels",size = 16 ))

    sns.distplot(np.array(predictions)[:,0]-y_test[0],label="Predicted Ionized Pixels {}".format(str(y_test[0])))
    sns.distplot(np.array(predictions)[:,1]-y_test[1],label="Predicted Ionized Pixels {}".format(str(y_test[1])))
    sns.distplot(np.array(predictions)[:,2]-y_test[2],label="Predicted Ionized Pixels {}".format(str(y_test[2])))
    sns.distplot(np.array(predictions)[:,3]-y_test[3],label="Predicted Ionized Pixels {}".format(str(y_test[3])))
    sns.distplot(np.array(predictions)[:,4]-y_test[4],label="Predicted Ionized Pixels {}".format(str(y_test[4])))

    plt.legend(markerscale=2.5)


def plot_pred_distribution_range(predictions, y_test, upper_lim, lower_lim):
    plt.figure(figsize=(15, 10))
    sns.distplot(np.array(predictions)[: , np.logical_and(y_test < upper_lim, y_test >= lower_lim)])
    plt.legend(markerscale=2.5)

"""
class coef_analysis:

    def __init__(self, y, x, labelname):
        self.y = y
        self.x = x
        self.labelname = labelname

    def estimate_coef(x, y):
        # number of observations/points
        n = np.size(x)

        # mean of x and y vector
        m_x, m_y = np.mean(x), np.mean(y)

        # calculating cross-deviation and deviation about x
        SS_xy = np.sum(y*x) - n*m_y*m_x
        SS_xx = np.sum(x*x) - n*m_x*m_x

        # calculating regression coefficients
        b_1 = SS_xy / SS_xx
        b_0 = m_y - b_1*m_x

        return [b_0, b_1]
        
    def plot_regression(x, y, b, labelname):
        # plotting the actual points as scatter plot
        plt.figure(figsize=(15,15))
        plt.title(labelname)
        plt.scatter(x,y, s=6, lw=0, alpha=0.9)

        # predicted response vector
        y_pred = b[0] + b[1]*x

        # plotting the regression line
        plt.plot(x, y_pred, 'k--',lw=5,alpha=1.0)
        
        # Plot trendline
        xx = np.linspace(0.95*np.min(x), 1.05*np.max(x), 1000)
        plt.plot(xx, xx, 'r--',lw=5,alpha=1.0)

        # putting labels
        plt.xlabel('True Optical Depth', size = 16)
        plt.ylabel('Prediction', size=16)
        
        #plt.legend(markerscale=2.5)
        plt.tight_layout()

        # Save plot
        #plt.savefig(outputdir+"residual_{}.png".format(str(fold)))
        #plt.clf()

    # Plot Slope and y-intercept distribution
    def statistics_plot_coef(coefficient):
        coefficients = np.array(coefficient,dtype=np.float64)
        plt.plot(np.arange(1,11),coefficients,'o')
        mean_value = np.mean(coefficients,axis=0)
        mean = np.zeros_like(coefficients)
        mean[:,0]=mean_value[0]
        mean[:,1]=mean_value[1]
        plt.plot(np.arange(1,11),mean,'k--',lw=1,markersize=4)
        plt.text(5.0, 0.2, r'$\mu$ vector [b_0 , b_1]: '+str(mean_value),
                {'color': 'blue', 'fontsize': 10, 'ha': 'center', 'va': 'center',
                'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})
                
        plt.xlabel("Fold Number")
        plt.ylabel("Distance from Mean")
        #plt.savefig(outputdir+"bestmodel_residual_of_mean.png")
        plt.clf()

    # Plot values of the deviation from the mean
    def statistics_plot_dev(coefficient):
        coefficients = np.array(coefficient,dtype=np.float64)

        plt.figure(figsize=(13,8))
        plt.plot(np.arange(1,11),coefficients,'o')
        mean_value = np.mean(coefficients,axis=0)
        mean = np.zeros_like(coefficients)
        mean[:,0]=mean_value[0]
        mean[:,1]=mean_value[1]
        plt.plot(np.arange(1,11),mean,'k--',lw=1,markersize=4)
        plt.text(5.0, 0.2, r'$\mu$ vector [b_0 , b_1]: '+str(mean_value),
                {'color': 'blue', 'fontsize': 10, 'ha': 'center', 'va': 'center',
                'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})
        for indx, coef in enumerate(coefficients[:,1]):
            diff = (coef-mean_value[1])/2
            plt.vlines(indx+1,coef,mean_value[1],colors='r',linestyles= 'dashed')
            plt.text(indx+1.25, coef-diff, r'%.2f' % diff,
            {'color': 'red', 'fontsize': 10, 'ha': 'center', 'va': 'center'})

        plt.xlabel("Fold Number")
        plt.ylabel("Distance from Mean")
        #plt.savefig(outputdir+"bestmodel_residual_of_mean_dev.png")
        plt.clf()

    def main(x, y, labelname):
        # Read in observations
        #x = read_results(filename)[0]
        #y = read_results(filename)[1]

        # estimating coefficients
        b = estimate_coef(x, y)
        coefficients.append(b)
        print("Estimated coefficients:\nb_0 = {} \
            \nb_1 = {}".format(b[0], b[1]))
            
        # write the coefficients to some text file
        fp.write(str(b)+"\n")

        # plotting regression line
        plot_regression(x, y, b, labelname)
        return
"""



def plot_qm_qs_vals(bnn_model):
    
    names = [layer.name for layer in bnn_model.layers if "flipout" in layer.name]
    
    qm_vals = [
        layer.kernel_posterior.mean()
        for layer in bnn_model.layers
        if "flipout" in layer.name
    ]
    qs_vals = [
        layer.kernel_posterior.stddev()
        for layer in bnn_model.layers
        if "flipout" in layer.name
    ]


    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 2, 1)
    for n, qm in zip(names, qm_vals):
        sns.distplot(tf.reshape(qm, shape=[-1]), ax=ax, label=n)
    ax.set_title("weight means",size = 16 )
    ax.set_xlim([-1.5, 1.5])
    ax.legend()
    ax = fig.add_subplot(1, 2, 2)
    for n, qs in zip(names, qs_vals):
        sns.distplot(tf.reshape(qs, shape=[-1]), ax=ax)
    ax.set_title("weight stddevs", size = 16)
    #ax.set_xlim([0, 1.0])
    fig.tight_layout()
    plt.show()
 

def plot_compare_pred_values(yhat, predictions, y_test):
    plt.figure(figsize=(13,13))
    plt.plot(yhat.mean(), np.mean(np.array(predictions), axis=0), 'r o', label = "")
    plt.plot(y_test,y_test, "k--", linewidth=5)

    plt.xlabel("DistributionLambda Mean", size = 16)
    plt.ylabel("MC Mean", size = 16)
    plt.legend(markerscale=2.5)
    plt.tight_layout()

def plot_compare_STDs(yhat, predictions, y_test):
    plt.figure(figsize=(15,15))
    plt.plot(yhat.stddev(), np.std(np.array(predictions), axis=0), 'r o', label = "")
    plt.plot(y_test/55,y_test/55, "k--", linewidth=5)

    plt.xlabel("DistributionLambda STD", size = 16)
    plt.ylabel("MC STD", size = 16)
    plt.legend(markerscale=2.5)
    plt.tight_layout()

def plot_one_to_one_pred_MC_DistributionLambda(yhat, predictions, y_test):
    plt.figure(figsize=(15,15))
    plt.title("Monte Carlo vs Sigma in output layer", size = 16)

    plt.plot(y_test, np.mean(np.array(predictions), axis=0), 'r o', alpha = 0.5, label = "Method 1: Monte Carlo Sample")
    plt.plot(y_test, yhat.mean(), 'c *', alpha = 0.5, label = "Method 2: DistributionLambda")
    plt.plot(y_test,y_test, "k--", linewidth=5)

    plt.xlabel("Ionized Pixels", size = 16)
    plt.ylabel("Predictions", size = 16)
    plt.legend(markerscale=2.5)
    plt.tight_layout()

def plot_error1(yhat, predictions, y_test):
    plt.figure(figsize=(15,15))

    plt.errorbar(y_test[0::20], np.mean(np.array(predictions), axis=0)[0::20], yerr=np.std(predictions, axis=0)[0::20], marker='.',
                 mfc='orangered', mec='green', c='green', mew=2.5, ms=20, ls='', alpha = 0.9, label="Method 1: Monte Carlo Sample")
    plt.errorbar(y_test[0::20], yhat.mean().numpy()[0::20,0], yerr=yhat.stddev().numpy()[0::20,0], marker='.',
                 mfc='lightpink', mec='blue', c='blue', mew=2.5, ms=20, ls='', alpha = 0.4, label="Method 2: DistributionLambda")

    plt.plot(y_test[0::20],np.mean(np.array(predictions), axis=0)[0::20]+np.std(predictions, axis=0)[0::20], 'g-', alpha=0.5,
             label='mean + 1 stddev'
            )
    plt.plot(y_test[0::20],np.mean(np.array(predictions), axis=0)[0::20]-np.std(predictions, axis=0)[0::20], 'g-', alpha=0.5,
             label='mean - 1 stddev'
            )

    plt.plot(y_test[0::20],yhat.mean().numpy()[0::20]+yhat.stddev().numpy()[0::20], '-',color='midnightblue', alpha=0.25,
             label='mean + 1 stddev'
            )
    plt.plot(y_test[0::20],yhat.mean().numpy()[0::20]-yhat.stddev().numpy()[0::20], '-',color='midnightblue', alpha=0.25,
             label='mean - 1 stddev'
            )

    plt.plot(y_test,y_test, "k--", linewidth=7)
    plt.xlabel("Ionized Pixels", size = 16)
    plt.ylabel("Predicitons", size = 16)
    plt.legend(markerscale=3.5)
    plt.tight_layout()

def plot_error2(yhat, predictions, y_test):

    f=plt.figure(figsize=(15,15))

    ax = f.add_subplot(111)
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("right")

    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4, color='r')

    plt.title("Error bars for the posterior using Monte Carlo vs Sigma in output layer", size = 16)

    ax.errorbar(y_test[0::10], (np.mean(np.array(predictions), axis=0)-y_test)[0::10], yerr=np.std(predictions, axis=0)[0::10], marker='.',
                 mfc='red', mec='green', c='green', mew=2.5, ms=20, ls='', label="Method 1: Monte Carlo Sample")
    ax.errorbar(y_test[0::10], yhat.mean().numpy()[0::10,0]-y_test[0::10], yerr=yhat.stddev().numpy()[0::10,0], marker='.',
                 mfc='pink', mec='blue', c='blue', mew=2.5, ms=20, ls='', label="Method 2: DistributionLambda")

    plt.hlines(y=0, xmin=0, xmax=y_test.max(), colors='k', linestyles='solid', linewidth=6, label='')
    
    plt.xlabel("Ionized Pixels", size = 16)
    plt.ylabel("Predictons", size = 16)
    plt.legend(markerscale=2.5)
    plt.tight_layout()


def plot_distribution_mean_std(yhat, predictions, y_test):
    plt.figure(figsize=(15, 10))
    sns.distplot(yhat.mean(),label="DistributionLambda")
    sns.distplot(np.mean(np.array(predictions), axis=0), label= "MC")
    sns.distplot(y_test, label= "Test Data")
    plt.title("DenseFlipout mean",size = 16 )
    #plt.xlim([0, 1.25])
    #plt.legend()

    plt.figure(figsize=(15, 10))
    sns.distplot(yhat.stddev(),label="DistributionLambda")
    sns.distplot(np.std(np.array(predictions), axis=0), label= "MC")
    plt.title("DenseFlipout stddev",size = 16 )
    #plt.xlim([0, 1.25])
    plt.legend()

# Noise
#var_nomode = [.01, .001, .0001]
#var_mode = [0.01, 0.001, 1e-4 ]

def add_noise_to_data(variances, X_train, X_test):
    
    
    
    def noisy(noise_typ, image, var):

        """
        Parameters
        ----------
        image : ndarray
            Input image data. Will be converted to float.
        mode : str
            One of the following strings, selecting the type of noise to add:

            'gauss'     Gaussian-distributed additive noise.
            'poisson'   Poisson-distributed noise generated from the data.
            's&p'       Replaces random pixels with 0 or 1.
            'speckle'   Multiplicative noise using out = image + n*image,where
                        n is uniform noise with specified mean & variance.
        """

        if noise_typ == "gauss":
            row,col,ch= image.shape
            mean = 0
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = image + gauss
            return noisy
        elif noise_typ == "s&p":
            row,col,ch = image.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
            out[coords] = 0
            return out
        elif noise_typ == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy
        elif noise_typ =="speckle":
            row,col,ch = image.shape
            gauss = np.random.randn(row,col,ch)
            gauss = gauss.reshape(row,col,ch)
            noisy = image + image * gauss
            return noisy


    noisy_X_train = []
    noisy_X_test = []
    for variance in variances:
        print("Variance ",variance)
        # copying over data
        print("Reading in data ...")
        x_test = copy.deepcopy(X_test)
        x_train = copy.deepcopy(X_train)

        for i in range(x_test.shape[0]):
            # replace each image with noisy data
            x_test[i] = noisy(noise_typ="gauss",image= x_test[i], var=variance)

        for i in range(x_train.shape[0]):
            # replace each image with noisy data
            x_train[i] = noisy(noise_typ="gauss",image= x_train[i], var=variance)

        # append to list
        print("Append data ...")
        noisy_X_train.append(x_train)
        noisy_X_test.append(x_test)
           
    noisy_X_train = np.array(noisy_X_train)
    noisy_X_test = np.array(noisy_X_test)
    
    print("Training data shape:", noisy_X_train.shape)
    print("Test data shape:", noisy_X_test.shape)
    
    return noisy_X_train, noisy_X_test


