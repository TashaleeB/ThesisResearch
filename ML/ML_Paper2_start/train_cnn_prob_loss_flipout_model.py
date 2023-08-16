"""
This script is responisble the different model architectures.
"""

# Import libraries
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
factor = 100
num_models = 25

# Load and Normalizing the data
nx, ny, training_data, X_train, y_train, X_test, y_test = utils.load_zreion_data(inputFile=inputFile, factor = 1000)#load_data(trainig_data_name=trainig_data_name, 
                                                                #          path=PATH, file_name="/ocean/projects/ast180004p/tbilling/data/t21_snapshots_nowedge_v12.hdf5", factor=100 )#"GenLight_2000sim.hdf5", factor = 100)

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
 


############################################################
# Models Training
############################################################
# group these models together
pm1_nll = model_fn.model1_ploss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                  nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                  loss_name='neg_log_likelihood', model_name='pm1',
                                  path=PATH, filter=FILTER,kl_divergence_function=kl_divergence_function,
                                  trainig_data_name=trainig_data_name)
pm2_nll = model_fn.model2_ploss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                  nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                  loss_name='neg_log_likelihood', model_name='pm2',
                                  path=PATH, filter=FILTER,kl_divergence_function=kl_divergence_function,
                                  trainig_data_name=trainig_data_name)
pm3_nll = model_fn.model3_ploss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                  nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                  loss_name='neg_log_likelihood', model_name='pm3',
                                  path=PATH, filter=FILTER,kl_divergence_function=kl_divergence_function,
                                  trainig_data_name=trainig_data_name)
pm4_nll = model_fn.model4_ploss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                  nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                  loss_name='neg_log_likelihood', model_name='pm4',
                                  path=PATH, filter=FILTER,kl_divergence_function=kl_divergence_function,
                                  trainig_data_name=trainig_data_name)



# group these models together
pm1_elbo = model_fn.model1_ploss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                  nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                  loss_name='elbo', model_name='pm1',
                                  path=PATH, filter=FILTER,kl_divergence_function=kl_divergence_function,
                                  trainig_data_name=trainig_data_name)
pm2_elbo = model_fn.model2_ploss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                  nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                  loss_name='elbo', model_name='pm2',
                                  path=PATH, filter=FILTER,kl_divergence_function=kl_divergence_function,
                                  trainig_data_name=trainig_data_name)
pm3_elbo = model_fn.model3_ploss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                  nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                  loss_name='elbo', model_name='pm3',
                                  path=PATH, filter=FILTER,kl_divergence_function=kl_divergence_function,
                                  trainig_data_name=trainig_data_name)
pm4_elbo = model_fn.model4_ploss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                  nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                  loss_name='elbo', model_name='pm4',
                                  path=PATH, filter=FILTER,kl_divergence_function=kl_divergence_function,
                                  trainig_data_name=trainig_data_name)



############################################################
# Make Preditions
############################################################
mean_predictions_pm_nll, distrib_predictions_pm_nll = utils.mc_pred_tuner_models(trainig_data_name=trainig_data_name,
                                        tuner_models=np.array([pm1_nll, pm2_nll, pm3_nll, pm4_nll]),
                                        X_test=X_test, model_name='pm_nll', path=PATH,
                                        filter=FILTER)
mean_predictions_pm_elbo, distrib_predictions_pm_elbo = utils.mc_pred_tuner_models(trainig_data_name=trainig_data_name,
                                        tuner_models=np.array([pm1_elbo, pm2_elbo, pm3_elbo, pm4_elbo]),
                                        X_test=X_test, model_name='pm_elbo', path=PATH,
                                        filter=FILTER)

############################################################
# Make Plots
############################################################

# Flipout with Probabilistic Loss Functions
model_name_list_pm = ['pm_nll', 'pm_elbo']
avg_predictions_list_pm = [mean_predictions_pm_nll, mean_predictions_pm_elbo]
predictions_list_pm = [distrib_predictions_pm_nll, distrib_predictions_pm_elbo]

bnn_pm = [pm1_nll, pm2_nll, pm3_nll, pm4_nll,
         pm1_elbo, pm2_elbo, pm3_elbo, pm4_elbo]
bnn_pm_model_name_list = ["pm1_nll", "pm2_nll", "pm3_nll", "pm4_nll",
         "pm1_elbo", "pm2_elbo", "pm3_elbo", "pm4_elbo"]

yhat_predictions_dm = np.array([yhat(X_test) for yhat in np.array(bnn_pm)])
yhat_predictions_dm.reshape(2, 4).shape
yhat_predictions_dm = yhat_predictions_dm.reshape(2,4)
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

for i in range(len(predictions_list_pm)):
    # coompare MC-sampling and DistributionLambda
    plots.plot_one_to_one_pred_MC_DistributionLambda(trainig_data_name=trainig_data_name,
                        model_name=model_name_list_pm[i],
                        yhat=yhat_predictions_dm[i],
                        mc_predictions=np.array(predictions_list_pm)[i],
                        y_test=y_test, path=PATH, filter=FILTER)

for i in range(len(predictions_list_pm)):
    # plot the error
    plots.plot_error1(trainig_data_name=trainig_data_name,
                        model_name=model_name_list_pm[i],
                        yhat=yhat_predictions_dm[i],
                        mc_predictions=np.array(predictions_list_pm)[i],
                        y_test=y_test, path=PATH, filter=FILTER)
                
for i in range(len(predictions_list_pm)):
    # plot the error
    plots.plot_error2(trainig_data_name=trainig_data_name,
                        model_name=model_name_list_pm[i],
                        yhat=yhat_predictions_dm[i],
                        mc_predictions=np.array(predictions_list_pm)[i],
                        y_test=y_test, path=PATH, filter=FILTER)

for i in range(len(predictions_list_pm)):
    # plot mean and std
    plots.plot_distribution_mean_std(trainig_data_name=trainig_data_name,
                        model_name=model_name_list_pm[i],
                        yhat=yhat_predictions_dm[i],
                        mc_predictions=np.array(predictions_list_pm)[i],
                        y_test=y_test, path=PATH, filter=FILTER)
