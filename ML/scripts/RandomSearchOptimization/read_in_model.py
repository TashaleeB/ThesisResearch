#printf "\e[?2004l"
from __future__ import print_function, division, absolute_import

# Imports
import os, sys, matplotlib, h5py, random, glob, json
import numpy as np, matplotlib.pyplot as plt, tensorflow as tf

matplotlib.use('Agg')

from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras import backend as K
#import pandas as pd

#random seed to control the reproducability
seed = 8675309
np.random.seed(seed)
#np.random.seed(datetime.datetime.now().microsecond)
path = '/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/opt_versions/200steps/'
data_path = path+'opt_2DConv_dense_layer_200Steps/'
model_path = path
outputdir = model_path

def Mean_Squared_over_true_Error(y_true, y_pred):
    # Create a custom loss function that divides the difference by the true
    y_true = K.cast(y_true, y_pred.dtype) #Casts a tensor to a different dtype and returns it.
    diff_ratio = K.square((y_pred - y_true)/K.clip(K.abs(y_true),K.epsilon(),None))
    loss = K.mean(diff_ratio, axis=-1)
    # Return a function

    return loss

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

models = sorted(glob.glob(model_path+"hyperParam_model_*.h5"))

for m in models:
    #custom_objects={"r2_keras":r2_keras}
    model = load_model(m, compile=True,
    custom_objects={"r2_keras":r2_keras, "Mean_Squared_over_true_Error":Mean_Squared_over_true_Error})
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.),
                loss=Mean_Squared_over_true_Error,
                metrics=[r2_keras,'mse', 'mae', 'mape'])


# save as YAML
yaml_string = model.to_yaml()

# model reconstruction from YAML:
model = model_from_yaml(yaml_string)


# read json files
trial_files = glob.glob(data_path+"/trial_*/trial.json")

with open(trial_files[0]) as f:
  data = json.load(f)

bestmodel = load_model("__hyperParam_model.h5",custom_objects={"r2_keras":r2_keras,"Mean_Squared_over_true_Error":Mean_Squared_over_true_Error})

bestmodel.compile(optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.),
                  loss=Mean_Squared_over_true_Error,
                  metrics=[r2_keras,'mse', 'mae', 'mape'])
                  
np.array(list(bestmodel.layers[0].variables))[1]
score=bestmodel.evaluate(val_x,  val_y, batch_size=32, verbose=1)

# WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
