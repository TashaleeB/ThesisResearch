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
#from tensorflow.keras.datasets import mnist
#from tensorflow.keras.utils import to_categorical, plot_model
#from keras.callbacks import ModelCheckpoint#, TensorBoard
from keras import backend as K

#from keras_tuner.tuners import RandomSearch
#from keras_tuner.engine.hypermodel import HyperModel
#from keras_tuner.engine.hyperparameters import HyperParameters
#from tensorboard.plugins.hparams import api as hp
#from keras.utils.generic_utils import get_custom_objects


import gc
gc.enable()
warnings.filterwarnings("ignore")

#hp = HyperParameters()
#from livelossplot import PlotLossesKeras
#plotlosses = PlotLossesKeras()

# import module
sys.path.insert(1, '/Users/tashaleebillings/Desktop/Tasha_Desktop/ThesisResearch/ML/ML_Paper2_Scripts/')
import utilities as utils, model_functions as model_fn, plots as plots

"""
# load data
trainig_data_name="toymodel"
toy_model = np.load('toy_models_32x32.npz')#np.load('toy_models_32x32.npz')
nx, ny, ntrain = toy_model['training_data'].shape
training_data = toy_model['training_data'].T
labels = toy_model['labels']

outputdir = "./"
idx = str(nx)
FILTER = "nowedge"
PATH = "./"
project_name='toy_model_tuner_{}'.format(idx)
HP_EPOCH = 3
HP_BATCH_SIZE = 5
factor = 100
num_models = 25

# Normalizing the data
X_train = training_data[0:8000,:,:].reshape(8000,nx,ny,1)
y_train = labels[0:8000]*factor
X_test = training_data[8000:,:,:].reshape(2000,nx,ny,1)
y_test = labels[8000:]*factor

print("training", X_train.shape)
print("validation", X_test.shape)
"""
# Load data
trainig_data_name="21cmfast"#"zreion"                                                                                             \
                                                                                                                                   
outputdir = "./21cm/"
FILTER = "nowedge"
PATH = outputdir#"./"                                                                                                             \
                                                                                                                                   
HP_EPOCH = 3
HP_BATCH_SIZE = 5
factor = 100
num_models = 25

nx, ny, training_data, X_train, y_train, X_test, y_test = utils.load_data(trainig_data_name=trainig_data_name,
                path=PATH, file_name="GenLight_2000sim.hdf5", factor = 100)
                
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





# group these models together
dm1_mse = model_fn.model1_dloss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                              loss_name='mse', model_name='dm1',
                                              path=PATH, filter=FILTER,kl_divergence_function=kl_divergence_function,
                                              trainig_data_name=trainig_data_name)

dm2_mse = model_fn.model2_dloss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                              loss_name='mse', model_name='dm2',
                                              path=PATH, filter=FILTER,kl_divergence_function=kl_divergence_function,
                                              trainig_data_name=trainig_data_name)
dm3_mse = model_fn.model3_dloss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                              loss_name='mse', model_name='dm3',
                                              path=PATH, filter=FILTER,kl_divergence_function=kl_divergence_function,
                                              trainig_data_name=trainig_data_name)
dm4_mse = model_fn.model4_dloss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                              loss_name='mse', model_name='dm4',
                                              path=PATH, filter=FILTER,kl_divergence_function=kl_divergence_function,
                                              trainig_data_name=trainig_data_name)

# group these models together
dm1_mae = model_fn.model1_dloss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                              loss_name='mae', model_name='dm1',
                                              path=PATH, filter=FILTER,kl_divergence_function=kl_divergence_function,
                                              trainig_data_name=trainig_data_name)
dm2_mae = model_fn.model2_dloss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                              loss_name='mae', model_name='dm2',
                                              path=PATH, filter=FILTER,kl_divergence_function=kl_divergence_function,
                                              trainig_data_name=trainig_data_name)
dm3_mae = model_fn.model3_dloss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                              loss_name='mae', model_name='dm3',
                                              path=PATH, filter=FILTER,kl_divergence_function=kl_divergence_function,
                                              trainig_data_name=trainig_data_name)
dm4_mae = model_fn.model4_dloss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                              loss_name='mae', model_name='dm4',
                                              path=PATH, filter=FILTER,kl_divergence_function=kl_divergence_function,
                                              trainig_data_name=trainig_data_name)

# group these models together
dm1_msote = model_fn.model1_dloss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                              loss_name='Mean_Squared_over_true_Error', model_name='dm1',
                                              path=PATH, filter=FILTER,kl_divergence_function=kl_divergence_function,
                                              trainig_data_name=trainig_data_name)
dm2_msote = model_fn.model2_dloss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                              loss_name='Mean_Squared_over_true_Error', model_name='dm2',
                                              path=PATH, filter=FILTER,kl_divergence_function=kl_divergence_function,
                                              trainig_data_name=trainig_data_name)
dm3_msote = model_fn.model3_dloss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                              loss_name='Mean_Squared_over_true_Error', model_name='dm3',
                                              path=PATH, filter=FILTER,kl_divergence_function=kl_divergence_function,
                                              trainig_data_name=trainig_data_name)
dm4_msote = model_fn.model4_dloss(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              nx=nx, ny=ny, epochs=HP_EPOCH, batch_size=HP_BATCH_SIZE,
                                              loss_name='Mean_Squared_over_true_Error', model_name='dm4',
                                              path=PATH, filter=FILTER,kl_divergence_function=kl_divergence_function,
                                              trainig_data_name=trainig_data_name)



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

                                        
# make list of predictions using all of the model
mean_predictions_dm_mse, distrib_predictions_dm_mse = utils.mc_pred_tuner_models(trainig_data_name=trainig_data_name,
                                        tuner_models=np.array([dm1_mse, dm2_mse, dm3_mse, dm4_mse]),
                                        X_test=X_test, model_name='dm_mse', path=PATH,
                                        filter=FILTER)
mean_predictions_dm_mae, distrib_predictions_dm_mae = utils.mc_pred_tuner_models(trainig_data_name=trainig_data_name,
                                        tuner_models=np.array([dm1_mae, dm2_mae, dm3_mae, dm4_mae]),
                                        X_test=X_test, model_name='dm_mae', path=PATH,
                                        filter=FILTER)
mean_predictions_dm_msote, distrib_predictions_dm_msote = utils.mc_pred_tuner_models(trainig_data_name=trainig_data_name,
                                        tuner_models=np.array([dm1_msote, dm2_msote, dm3_msote, dm4_msote]),
                                        X_test=X_test, model_name='dm_msote', path=PATH,
                                        filter=FILTER)


# make list of predictions using all of the model
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

# Flipout with Deterministic Loss Functions
model_name_list = ['dm_mse', 'dm_mae', 'dm_msote']
avg_predictions_list = [mean_predictions_dm_mse, mean_predictions_dm_mae, mean_predictions_dm_msote]
predictions_list = [distrib_predictions_dm_mse, distrib_predictions_dm_mae, distrib_predictions_dm_msote]
#array_mc_predictions = np.array([distrib_predictions_dm_mse, distrib_predictions_dm_mae, distrib_predictions_dm_msote])

bnn_dm = [dm1_mse, dm2_mse, dm3_mse, dm4_mse,
         dm1_mae, dm2_mae, dm3_mae, dm4_mae,
         dm1_msote, dm2_msote, dm3_msote, dm4_msote]
bnn_dm_model_name_list = ["dm1_mse", "dm2_mse", "dm3_mse", "dm4_mse",
         "dm1_mae", "dm2_mae", "dm3_mae", "dm4_mae",
         "dm1_msote", "dm2_msote", "dm3_msote", "dm4_msote"]

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

for i in range(len(bnn_dm)):
    # plot distribution of the weights
    plots.plot_qm_qs_vals(trainig_data_name=trainig_data_name, model_name=bnn_dm_model_name_list[i],
                    bnn_model=bnn_dm[i], path=PATH, filter=FILTER)


for i in range(len(predictions_list)):
    # plot the error
    plots.plot_deterministic_model_error1(trainig_data_name=trainig_data_name,
                model_name=model_name_list[i],
                mc_predictions=np.array(predictions_list)[i],
                y_test=y_test, path=PATH, filter=FILTER)
                
for i in range(len(predictions_list)):
    # plot the error
    plots.plot_deterministic_model_error2(trainig_data_name=trainig_data_name,
                model_name=model_name_list[i],
                mc_predictions=np.array(predictions_list)[i],
                y_test=y_test, path=PATH, filter=FILTER)


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
