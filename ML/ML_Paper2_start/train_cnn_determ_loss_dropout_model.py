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
trainig_data_name="21cmfast"#"zreion"                                                                                                                                                         
outputdir = "./21cm/"
FILTER = "nowedge"
PATH = outputdir#"./"                                                                                                                                                                         
HP_EPOCH = 3
HP_BATCH_SIZE = 5
factor = 100
num_models = 25

# Load and Normalizing the data                                                                                                                                                               
nx, ny, training_data, X_train, y_train, X_test, y_test = utils.load_data(trainig_data_name=trainig_data_name,
                path=PATH, file_name="/ocean/projects/ast180004p/tbilling/data/t21_snapshots_nowedge_v12.hdf5", factor=100)#"GenLight_2000sim.hdf5", factor = 100)

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
dropoutm1_mse = model_fn.dropout_model1_dloss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                              loss_name='mse', model_name='dropoutm1',
                                              path=PATH, filter=FILTER,
                                              trainig_data_name=trainig_data_name, training = True)

dropoutm2_mse = model_fn.dropout_model2_dloss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                              loss_name='mse', model_name='dropoutm2',
                                              path=PATH, filter=FILTER,
                                              trainig_data_name=trainig_data_name, training = True)
dropoutm3_mse = model_fn.dropout_model3_dloss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                              loss_name='mse', model_name='dropoutm3',
                                              path=PATH, filter=FILTER,
                                              trainig_data_name=trainig_data_name, training = True)
dropoutm4_mse = model_fn.dropout_model4_dloss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                              loss_name='mse', model_name='dropoutm4',
                                              path=PATH, filter=FILTER,
                                              trainig_data_name=trainig_data_name, training = True)

# group these models together
dropoutm1_mae = model_fn.dropout_model1_dloss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                              loss_name='mae', model_name='dropoutm1',
                                              path=PATH, filter=FILTER,
                                              trainig_data_name=trainig_data_name, training = True)
dropoutm2_mae = model_fn.dropout_model2_dloss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                              loss_name='mae', model_name='dropoutm2',
                                              path=PATH, filter=FILTER,
                                              trainig_data_name=trainig_data_name, training = True)
dropoutm3_mae = model_fn.dropout_model3_dloss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                              loss_name='mae', model_name='dropoutm3',
                                              path=PATH, filter=FILTER,
                                              trainig_data_name=trainig_data_name, training = True)
dropoutm4_mae = model_fn.dropout_model4_dloss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                              loss_name='mae', model_name='dropoutm4',
                                              path=PATH, filter=FILTER,
                                              trainig_data_name=trainig_data_name, training = True)

# group these models together
dropoutm1_msote = model_fn.dropout_model1_dloss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                              loss_name='Mean_Squared_over_true_Error', model_name='dropoutm1',
                                              path=PATH, filter=FILTER,
                                              trainig_data_name=trainig_data_name, training = True)
dropoutm2_msote = model_fn.dropout_model2_dloss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                              loss_name='Mean_Squared_over_true_Error', model_name='dropoutm2',
                                              path=PATH, filter=FILTER,
                                              trainig_data_name=trainig_data_name, training = True)
dropoutm3_msote = model_fn.dropout_model3_dloss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                              loss_name='Mean_Squared_over_true_Error', model_name='dropoutm3',
                                              path=PATH, filter=FILTER,
                                              trainig_data_name=trainig_data_name, training = True)
dropoutm4_msote = model_fn.dropout_model4_dloss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                              loss_name='Mean_Squared_over_true_Error', model_name='dropoutm4',
                                              path=PATH, filter=FILTER,
                                              trainig_data_name=trainig_data_name, training = True)



############################################################
# Make Preditions
############################################################

# make list of predictions using all of the model
mean_predictions_dropoutm_mse, distrib_predictions_dropoutm_mse = utils.mc_pred_tuner_models(trainig_data_name=trainig_data_name,
                                        tuner_models=np.array([dropoutm1_mse, dropoutm2_mse, dropoutm3_mse, dropoutm4_mse]),
                                        X_test=X_test, model_name='dropoutm_mse', path=PATH,
                                        filter=FILTER)
mean_predictions_dropoutm_mae, distrib_predictions_dropoutm_mae = utils.mc_pred_tuner_models(trainig_data_name=trainig_data_name,
                                        tuner_models=np.array([dropoutm1_mae, dropoutm2_mae, dropoutm3_mae, dropoutm4_mae]),
                                        X_test=X_test, model_name='dropoutm_mae', path=PATH,
                                        filter=FILTER)
mean_predictions_dropoutm_msote, distrib_predictions_dropoutm_msote = utils.mc_pred_tuner_models(trainig_data_name=trainig_data_name,
                                        tuner_models=np.array([dropoutm1_msote, dropoutm2_msote, dropoutm3_msote, dropoutm4_msote]),
                                        X_test=X_test, model_name='dropoutm_msote', path=PATH,
                                        filter=FILTER)


                                                               
############################################################
# Make Plots
############################################################

# Dropout
model_name_list = ['dropoutm_mse', 'dropoutm_mae', 'dropoutm_msote']
avg_predictions_list = [mean_predictions_dropoutm_mse, mean_predictions_dropoutm_mae, mean_predictions_dropoutm_msote]
predictions_list = [distrib_predictions_dropoutm_mse, distrib_predictions_dropoutm_mae, distrib_predictions_dropoutm_msote]

for i in range(len(avg_predictions_list)):
    # Make one-to-one plots of the average predictions from MC-sample
    plots.single_plot_average_predictions(trainig_data_name=trainig_data_name,
                                          avg_predictions=avg_predictions_list[i],
                                          y_test=y_test, model_name=model_name_list[i],
                                          path=PATH, filter=FILTER)

for i in range(len(predictions_list)):
    # Make plots of the distribution of a few predictions
    plots.plot_pred_distribution(trainig_data_name=trainig_data_name,
                                  predictions=predictions_list[i],
                                  y_test=y_test, model_name=model_name_list[i],
                                  path=PATH, filter=FILTER)
