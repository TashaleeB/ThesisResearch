"""
This script is responisble the different model architectures.
"""

# Import libraries
from __future__ import print_function, division, absolute_import
import numpy as np, matplotlib.pyplot as plt, seaborn as sns #, keras_tuner as kt,
import tensorflow as tf
import sys, os, warnings, h5py
from tqdm import tqdm
import tensorflow_probability as tfp

tfd = tfp.distributions

#from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, Flatten, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from keras import backend as K


import gc
gc.enable()
warnings.filterwarnings("ignore")


# import module
sys.path.insert(1, '/ocean/projects/ast180004p/tbilling/sandbox/bayesian/mlpaper2/')
import utilities as utils, model_functions as model_fn, plots as plots


# load data
trainig_data_name="zreion"#"21cmfast"
inputFile = "/ocean/projects/ast180004p/tbilling/data/t21_snapshots_nowedge_v12.hdf5"
outputdir = "./zreion/"#"./21cm/"
FILTER = "nowedge"
PATH = outputdir#"./"
HP_EPOCH = 3
HP_BATCH_SIZE = 5
factor = 1000
num_models = 25
color = 30

# Load and Normalizing the data
nx, ny, training_data, X_train, y_train, X_test, y_test = utils.load_zreion_data(inputFile=inputFile, factor = factor)
print("training", X_train.shape)
print("validation", X_test.shape)

# load divegence normalization function
kl_divergence_function = utils.kl_divergence_normalization_term(TRAINING_DATA=training_data)

# make custom functions callable with keras
# these are functions I made in the utility script and I just make them callable here.
keras.utils.get_custom_objects().update({'Mean_Squared_over_true_Error':utils.custom_loss.Mean_Squared_over_true_Error,
                                         'neg_log_likelihood': utils.custom_loss.neg_log_likelihood,
                                         'kl_divergence':utils.custom_loss.kl_divergence,
                                         'elbo':utils.custom_loss.elbo})
 
# Define Error Plot Functions
def predictions1(trainig_data_name, model_name, yhat, mc_predictions, y_test, path, filter="nowedge"):
    # Plot Error
    plt.figure(figsize=(15,15))
    plt.errorbar(y_test[0::10], np.mean(mc_predictions, axis=0)[0::10], yerr=np.std(mc_predictions, axis=0)[0::10], marker='.',
                         mfc='orangered', mec='green', c='green', mew=2.5, ms=20, ls='', alpha = 0.9, label="Method 1: Monte Carlo Sample")
    plt.errorbar(y_test[0::10], yhat.mean().numpy()[0::10,0], yerr=yhat.stddev().numpy()[0::10,0], marker='.',
                         mfc='lightpink', mec='blue', c='blue', mew=2.5, ms=20, ls='', alpha = 0.4, label="Method 2: DistributionLambda")
    plt.plot(y_test[0::10],np.mean(mc_predictions, axis=0)[0::10]+np.std(mc_predictions, axis=0)[0::10], 'r-', alpha=0.5,
             label='mean + 1 stddev'
            )
    plt.plot(y_test[0::10],np.mean(mc_predictions, axis=0)[0::10]-np.std(mc_predictions, axis=0)[0::10], 'r-', alpha=0.5,
             label='mean - 1 stddev'
            )
    plt.plot(y_test[0::10],yhat.mean().numpy()[0::10]+yhat.stddev().numpy()[0::10], 'g-', alpha=0.5,
             label='mean + 1 stddev'
            )
    plt.plot(y_test[0::10],yhat.mean().numpy()[0::10]-yhat.stddev().numpy()[0::10], 'g-', alpha=0.5,
             label='mean - 1 stddev'
            )
    plt.plot(y_test,y_test, "k--", linewidth=7, alpha=1.0)
    plt.xlabel(r"$\tau_{True}$", fontsize=16)
    plt.ylabel("Predicitons", size = 16)


    plt.legend(markerscale=1.5, fontsize='large')
    plt.tight_layout()

    
    # save plots as png and pdf
    plt.savefig(os.path.join(path, filter,
        "plot_error1_{}_{}.png".format(trainig_data_name, model_name)))
    plt.savefig(os.path.join(path, filter,
        "plot_error1_{}_{}.pdf".format(trainig_data_name, model_name)))
    
    return


def predictions2(trainig_data_name, model_name, yhat, mc_predictions, y_test, path, filter="nowedge"):
    # Plot Error
    plt.figure(figsize=(15,15))
    plt.errorbar(y_test[0::10], (np.mean(mc_predictions, axis=0)-y_test)[0::10], yerr=np.std(mc_predictions, axis=0)[0::10], marker='.',
                         mfc='orangered', mec='green', c='green', mew=2.5, ms=20, ls='', alpha = 0.9, label="Method 1: Monte Carlo Sample")
    plt.errorbar(y_test[0::10], yhat.mean().numpy()[0::10,0]-y_test[0::10], yerr=yhat.stddev().numpy()[0::10,0], marker='.',
                         mfc='lightpink', mec='blue', c='blue', mew=2.5, ms=20, ls='', alpha = 0.4, label="Method 2: DistributionLambda")
    plt.hlines(y=0, xmin=0, xmax=y_test.max(), colors='k', linestyles='solid', linewidth=6, label='')
    plt.xlabel(r"$\tau_{True}$", fontsize=16)
    plt.ylabel("Predicitons", size = 16)


    plt.legend(markerscale=1.5, fontsize='large')
    plt.tight_layout()

    
    # save plots as png and pdf
    plt.savefig(os.path.join(path, filter,
        "plot_error2_{}_{}.png".format(trainig_data_name, model_name)))
    plt.savefig(os.path.join(path, filter,
        "plot_error2_{}_{}.pdf".format(trainig_data_name, model_name)))
    
    return


############################################################
# Models Training
############################################################
# group these models together
pm1_msote = model_fn.old_model1(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                  nx=nx, ny=ny, color=color, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                  loss_name='Mean_Squared_over_true_Error', model_name='old_model1_pm',
                                  path=PATH, filter=FILTER,kl_divergence_function=kl_divergence_function,
                                  trainig_data_name=trainig_data_name, wedge=False, training=False)


# group these models together
pm2_nll = model_fn.old_model2(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                  nx=nx, ny=ny, color=color, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                  loss_name='neg_log_likelihood', model_name='old_model2_pm',
                                  path=PATH, filter=FILTER,kl_divergence_function=kl_divergence_function,
                                  trainig_data_name=trainig_data_name, wedge=False, training=False)
pm2_elbo = model_fn.old_model2(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                  nx=nx, ny=ny, color=color, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                  loss_name='elbo', model_name='old_model2_pm',
                                  path=PATH, filter=FILTER,kl_divergence_function=kl_divergence_function,
                                  trainig_data_name=trainig_data_name, wedge=False, training=False)




############################################################
# Make Preditions
############################################################
mean_predictions_old_model1_pm, distrib_predictions_old_model1_pm = utils.mc_pred_tuner_models(trainig_data_name=trainig_data_name,
                                        tuner_models=np.array([pm1_msote]), y_test=y_test,
                                        X_test=X_test, model_name='old_model_pm_msote', path=PATH,
                                        filter=FILTER)
mean_predictions_old_model2_pm, distrib_predictions_old_model2_pm = utils.mc_pred_tuner_models(trainig_data_name=trainig_data_name,
                                        tuner_models=np.array([pm2_nll, pm2_elbo,]), y_test=y_test,
                                        X_test=X_test, model_name='old_model_pm_nll_elbo', path=PATH,
                                        filter=FILTER)



############################################################
# Make Plots
############################################################

# Flipout with Probabilistic Loss Functions
model_name_list_pm = ['pm1_msote', 'pm2_nll', 'pm2_elbo']
avg_predictions_list_pm = [mean_predictions_old_model1_pm, mean_predictions_old_model2_pm]
predictions_list_pm = [distrib_predictions_old_model1_pm, distrib_predictions_old_model2_pm]

bnn_pm = [pm1_msote, pm2_nll, pm2_elbo]
bnn_pm_model_name_list = ["pm1_msote", "pm2_nll", "pm2_elbo"]

yhat_predictions_dm = np.array([yhat(X_test) for yhat in np.array(bnn_pm)])
print(yhat_predictions_dm)

for i in range(len(avg_predictions_list_pm)):
    # Make one-to-one plots of the average predictions from MC-sample
    plots.single_plot_average_predictions(trainig_data_name=trainig_data_name,
                                          avg_predictions=avg_predictions_list_pm[i],
                                          y_test=y_test, model_name=model_name_list_pm[i],
                                          path=PATH, filter=FILTER)

for i in range(len(predictions_list_pm)):
    # Make plots of the distribution of a few predictions
    plots.plot_pred_distribution(trainig_data_name=trainig_data_name,
                                  predictions=predictions_list_pm[i],
                                  y_test=y_test, model_name=model_name_list_pm[i],
                                  path=PATH, filter=FILTER)

for i in range(len(bnn_pm)):
    # plot distribution of the weights
    plots.plot_qm_qs_vals(trainig_data_name=trainig_data_name,
                        model_name=bnn_pm_model_name_list[i],
                        bnn_model=bnn_pm[i], path=PATH, filter=FILTER)




for i in range(len(distrib_predictions_old_model1_pm)):
    # plot the error
    predictions1(trainig_data_name=trainig_data_name,
                        model_name=model_name_list_pm[1:][i],
                        yhat=yhat_predictions_dm[1:][i],
                        mc_predictions=distrib_predictions_old_model2_pm[i],
                        y_test=y_test, path=PATH, filter=FILTER)


for i in range(len(distrib_predictions_old_model2_pm)):
    # plot the error
    predictions2(trainig_data_name=trainig_data_name,
                        model_name=model_name_list_pm[1:][i],
                        yhat=yhat_predictions_dm[1:][i],
                        mc_predictions=distrib_predictions_old_model2_pm[i],
                        y_test=y_test, path=PATH, filter=FILTER)
