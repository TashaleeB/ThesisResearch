# Import libraries
from __future__ import print_function, division, absolute_import
import numpy as np, matplotlib.pyplot as plt, seaborn as sns #, keras_tuner as kt,
import tensorflow as tf
import sys, os, warnings, h5py

import tensorflow_probability as tfp
tfd = tfp.distributions

#from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, Flatten, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import KFold

from keras import backend as K

from tqdm import tqdm

# Hyperparameter Tuning
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hypermodel import HyperModel
from keras_tuner.engine.hyperparameters import HyperParameters

import gc
gc.enable()
warnings.filterwarnings("ignore")

#random seed to control the reproducability
seed = 8675309
np.random.seed(seed)

# import module
sys.path.insert(1, "/ocean/projects/ast180004p/tbilling/scripts/")
import utilities as utils, model_functions as model_fn, plots as plots


def plot_error2(model_name, yhat, mc_predictions, y_test, training_data_name, loss_name):
    #fig, axes = plt.subplots(figsize=(15,10),nrows=1, ncols=1, sharex=False, sharey=False)
    plt.figure(figsize=(15,10))
    
    #plt.yaxis.tick_left()
    #plt.set_label_position("right")

    #plt.tick_params(which='both', width=2)
    #plt.tick_params(which='major', length=7)
    #plt.tick_params(which='minor', length=4, color='r')

    # plot error
    plt.errorbar(y_test[0::20], (np.mean(mc_predictions, axis=0)-y_test)[0::20], yerr=np.std(mc_predictions, axis=0)[0::20], marker='.', mfc='red', mec='green', c='green', mew=2.5, ms=20, ls='', label="Method 1: Monte Carlo Sample")

    #plt.errorbar(y_test[0::20], yhat.mean().numpy()[0::20,0]-y_test[0::20], yerr=yhat.stddev().numpy()[0::20,0], marker='.', mfc='pink', mec='blue', c='blue', mew=2.5, ms=20, ls='', label="Method 2: DistributionLambda")

    plt.hlines(y=0, xmin=0, xmax=y_test.max(), colors='k', linestyles='solid', linewidth=6, label='')#alpha=1.0)

    plt.hlines(y=2., xmin=0, xmax=y_test.max(), color="black", linewidth=6,
               linestyle="--", alpha=0.5, #label="CMB EE cosmic variance"
              )
    plt.hlines(y=-2., xmin=0, xmax=y_test.max(), color="black", linewidth=6,
               linestyle="--", alpha=0.5
              )
    # plot box plot
    plt.ylabel("Predicted", fontsize=16)

    plt.xlabel("Ionized Pixels", fontsize=16)

    plt.legend(markerscale=1.5, fontsize='large')
    plt.tight_layout()
    
    # save plots as png and pdf
    plt.savefig(os.path.join(PATH, FILTER, "plot_error2_{}_{}_{}.png".format(training_data_name, model_name, loss_name)))
    plt.savefig(os.path.join(PATH, FILTER, "plot_error2_{}_{}_{}.pdf".format(training_data_name, model_name, loss_name)))
        
    return


def plot_compare_STDs(yhat, predictions, y_test, model_name, training_data_name, loss_name):
    plt.figure(figsize=(15,15))
    plt.plot(yhat.stddev(), np.std(np.array(predictions), axis=0), 'r o', label = "")
    plt.plot(y_test/55,y_test/55, "k--", linewidth=5)

    plt.xlabel("DistributionLambda STD", size = 16)
    plt.ylabel("MC STD", size = 16)
    
    #plt.xlim(0.5,1.5)
    #plt.ylim(0,3)
    
    plt.legend(markerscale=2.5)
    plt.tight_layout()
    
    # save plots as png and pdf
    plt.savefig(os.path.join(PATH, FILTER, "plot_std_mc_lambda_{}_{}_{}.png".format(training_data_name, model_name, loss_name)))
    plt.savefig(os.path.join(PATH, FILTER, "plot_std_mc_lambda_{}_{}_{}.pdf".format(training_data_name, model_name, loss_name)))
    
    return



# load data
training_data_name="zreion_crop_256"#"21cmfast"
inputFile = "/ocean/projects/ast180004p/tbilling/data/t21_snapshots_nowedge_v12.hdf5"
outputdir = "./zreion/"#"./21cm/"
FILTER = "nowedge"
PROJECT_NAME="hp_opt"
PATH = outputdir#"./"
STEPS = 20
NFOLD = 10
factor = 1000
num_models = 20 # max number of models to train
loss_name = 'neg_log_likelihood'#'elbo'
model_name = 'flipout'
TRAINABLE = False
CROP_LENGTH = 256

# Load and Normalizing the data
nx, ny, training_data, X_train, y_train, X_test, y_test = utils.load_zreion_data(inputFile=inputFile, factor = factor)
color = X_train.shape[-1]
print("training", X_train.shape)
print("validation", X_test.shape)

# Crop the imagees for training and test image
x_crop = utils.crop(X_train[:640], img_height=X_train.shape[1], img_width=X_train.shape[2], crop_length=CROP_LENGTH)
y_crop = y_train[:640]
val_x_crop = utils.crop(X_train[640:800], img_height=X_train.shape[1], img_width=X_train.shape[2], crop_length=CROP_LENGTH)
val_y_crop = y_train[640:800]

x_test_crop = utils.crop(X_test, img_height=X_test.shape[1], img_width=X_test.shape[2], crop_length=256)


# load divegence normalization function
kl_divergence_function = utils.kl_divergence_normalization_term(TRAINING_DATA=training_data)

# make custom functions callable with keras
# these are functions I made in the utility script and I just make them callable here.
keras.utils.get_custom_objects().update({'Mean_Squared_over_true_Error':utils.custom_loss.Mean_Squared_over_true_Error,
                                         'neg_log_likelihood': utils.custom_loss.neg_log_likelihood,
                                         'kl_divergence':utils.custom_loss.kl_divergence,
                                         'elbo':utils.custom_loss.elbo,
                                         'r2_keras':utils.custom_loss.r2_keras,
                                        })

# initialization hyperparameter function
hp = HyperParameters()

# check to see if path exists
if os.path.isdir(PROJECT_NAME):
    os.system("rm -rvf {}".format(PROJECT_NAME))

def build_model(hp):
    
    input_shape=(CROP_LENGTH, CROP_LENGTH, color)
    input0 = Input(shape=input_shape)
    inner = tfp.layers.Convolution2DFlipout(filters = 16,
                                              kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , #padding = 'valid',
                                              )(input0)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    
    inner = tfp.layers.Convolution2DFlipout(filters = 32,
                                              kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , #padding = 'valid',
                                              )(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    
    inner = tfp.layers.Convolution2DFlipout(filters = 64,
                                              kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , #padding = 'valid',
                                              )(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)
    
    # Number of hidden layers
    for i in range(hp.Int('num_layers', 0, 2)):
        #Note that we still test a different number of units for each layer.
        #There is a requirement that each Hyperparameter name should be unique.
        inner = tfp.layers.Convolution2DFlipout(filters = hp.Int('filters_' + str(i),
                                                                 min_value=64,
                                                                 max_value=256,
                                                                 step=64),
                                              kernel_divergence_fn=kl_divergence_function,
                                              kernel_size=3, activation = 'relu'
                                              , #padding = 'valid',
                                              )(inner)
        inner = BatchNormalization()(inner)
        inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inner)
        
    inner = GlobalAveragePooling2D()(inner)
    
    inner = Dropout(0.2, trainable=TRAINABLE)(inner)
    inner = tfp.layers.DenseFlipout(units=hp.Int('units',min_value=100,max_value=300,step=100,default=100), activation='relu', kernel_divergence_fn=kl_divergence_function,)(inner)
    
    inner = Dropout(0.2, trainable=TRAINABLE)(inner)
    inner = tfp.layers.DenseFlipout(100, activation='relu', kernel_divergence_fn=kl_divergence_function,)(inner)
    
    inner = Dropout(0.2, trainable=TRAINABLE)(inner)
    inner = tfp.layers.DenseFlipout(20, activation='relu', kernel_divergence_fn=kl_divergence_function,)(inner)
    
    output = tfp.layers.DenseFlipout(1, activation='linear', kernel_divergence_fn=kl_divergence_function,)(inner)
    output = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1],
                           #scale = 1)))
                           scale=1e-3 + tf.math.softplus(0.01 * t[..., :1])))(output)
    
    model = Model(inputs=input0, outputs=output)
    
    # Compile Model
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.),
                  loss=loss_name,
                  metrics=['mse'])#=['r2_keras','mse', 'elbo','neg_log_likelihood', 'Mean_Squared_over_true_Error'])
    
    print(model.summary())
    
    return model

# Start Tuning based on low validation loss
tuner = RandomSearch(build_model,
                     objective='val_loss', # 'loss', 'val_loss', 'val_accuracy'
                     max_trials=num_models,
                     executions_per_trial=1,
                     directory=outputdir,
                     project_name=PROJECT_NAME)

# Print a summary of the search space
tuner.search_space_summary()

# Start Searching parameter space
tuner.search(x_crop, y_crop, epochs=STEPS, validation_data=(val_x_crop, val_y_crop))
tuner.results_summary()

# Retrieve the best model.
best_model = tuner.get_best_models(num_models=1)[0]

# Check to see if path to folder exists and if not create it
MYDIR = os.path.join(PATH, FILTER)
CHECK_FOLDER = os.path.isdir(MYDIR)

if not CHECK_FOLDER:
    os.makedirs(MYDIR)
    print("Created folder : ", MYDIR)

else:
    print(MYDIR, "Folder already exists.")

# Save best hyperperam Model
print("saving best hyperparameter model ...")
best_model.save(os.path.join(PATH, FILTER,"best_model_{}_{}_{}.h5".format(training_data_name, model_name, loss_name)))

# Save best hyperperam Model Weights
print("saving best hyperparameter model weights ...")
best_model.save_weights(os.path.join(PATH, FILTER,"best_model_weights_{}_{}_{}.h5".format(training_data_name, model_name, loss_name)))

# Save all other models so that we can see how the model behaves based on complexity
for indx in np.arange(len(tuner.get_best_models(num_models=num_models))):
    model_ = tuner.get_best_models(num_models=num_models)[indx]
    model_.save(os.path.join(PATH, FILTER,"best_model_{}_{}_{}_{}.h5".format(training_data_name, model_name, loss_name, str(indx+1))))
    print("Saving ... "+os.path.join(PATH, FILTER,"best_model_{}_{}_{}_{}.h5".format(training_data_name, model_name, loss_name, str(indx+1))))
    model_.save_weights(os.path.join(PATH, FILTER,"best_model_weights_{}_{}_{}_{}.h5".format(training_data_name, model_name, loss_name, str(indx+1))))
    print("Saving ... "+os.path.join(PATH, FILTER,"best_model_weights_{}_{}_{}_{}.h5".format(training_data_name, model_name, loss_name, str(indx+1))))

###### MAKE PREDICTIONS ######
print("Prediction of mean and distribution")
mean_predictions_best_model, distrib_predictions_best_model = utils.mc_pred_tuner_models(
    training_data_name=training_data_name,
    tuner_models=np.array([best_model]), y_test=y_test,
    X_test=x_test_crop, model_name=model_name, path=PATH,
    filter=FILTER,
)
pred_best_model = utils.mc_pred(
    training_data_name=training_data_name,
    bestmodel=best_model,
    X_test=x_test_crop,
    y_test=y_test,
    model_name=model_name,
    path=PATH,
    filter=FILTER,
)



###### MAKE IMAGES ######
print("one-to-one plots")
# Make one-to-one plots of the average predictions from MC-sample
plots.single_plot_average_predictions(trainig_data_name=trainig_data_name,
                                          avg_predictions=np.array([mean_predictions_best_model]),
                                          y_test=y_test, model_name=model_name,
                                          path=PATH, filter=FILTER)

print("Make plots of the distribution of a few predictions")
# Make plots of the distribution of a few predictions
plots.plot_pred_distribution(trainig_data_name=trainig_data_name,
                                  predictions=np.array([distrib_predictions_best_model]),
                                  y_test=y_test, model_name=model_name,
                                  path=PATH, filter=FILTER)
                                  
print("plot distribution of the weights")
# plot distribution of the weights
plots.plot_qm_qs_vals(trainig_data_name=trainig_data_name, model_name=model_name,
                    bnn_model=best_model, path=PATH, filter=FILTER)
                    
# Error Plots comparing MC-Sample and Lambda layer
plot_error2(mc_predictions= np.array(mean_predictions_best_model), yhat=best_model(x_test_crop), y_test=y_test, model_name=model_name, training_data_name=training_data_name)
plot_compare_STDs(yhat=best_model(x_test_crop), predictions= mean_predictions_best_model, y_test= y_test, model_name=model_name, training_data_name=training_data_name)


plt.figure(figsize=(15,10))
plt.hlines(y=0, xmin=0, xmax=y_test.max(), colors='k', linestyles='solid', linewidth=6, label='')
plt.plot(y_test[0::10], np.std(np.array(mean_predictions_best_model), axis=0)[0::10], "<", label="nll mc sample")
plt.plot(y_test[0::10], best_model(x_test_crop).stddev().numpy()[0::10,0], "^", label="nll lambda")
plt.xlabel("Label")
plt.ylabel("STDD")
plt.legend(markerscale=1.5, fontsize='large')
plt.savefig(os.path.join(PATH, FILTER, "error_as_function_of_input_{}_{}_{}.pdf".format(training_data_name, model_name, loss_name)))
plt.savefig(os.path.join(PATH, FILTER, "error_as_function_of_input_{}_{}_{}.png".format(training_data_name, model_name, loss_name)))




"""
kfold_split = sorted(glob.glob("train_test_index_*.npz"))
scores =[]

for i in np.arange(NFOLD):
    # Load Index Label
    train_index = np.load(kfold_split[i])["train_index"]
    test_index = np.load(kfold_split[i])["test_index"]
    # Load Labels and Images
    trainlabels = readLabels(ind=None)[train_index,5]*factor
    trainlabels = trainlabels.reshape(-1, 1)
    images,shape_label = readImages(ind=train_index)

    testlabels = readLabels(ind=None)[test_index,5]*factor
    testlabels = testlabels.reshape(-1, 1)
    test_image,input_shape = readImages(ind=test_index)

    fold = i
    log_dir = os.path.join(outputdir, 'output', str(fold))
    cb = keras.callbacks.TensorBoard(log_dir=log_dir,
                                     histogram_freq=10, write_images=True)


    print('making model...')
    print('compiling model...')
    print('number of regression parameters: ', trainlabels.shape[1])
    model = makeModel(input_shape, Nregressparams=trainlabels.shape[1])

    #model = load_model(outputdir+'hyperParam_model.h5')
    #model = load_model(outputdir+'hyperParam_model.h5',custom_objects={"r2_keras":r2_keras})

    # The fit() method - Trains the model with the given inputs (and corresponding training labels)
    print('fitting model...')
    print("-"*150)
    print("*"*150)
    print("-"*150)
    #print start time
    os.system("date")
    history = model.fit(images, trainlabels, batch_size=32, verbose=2,
                        validation_split=0.2, epochs=STEPS, callbacks=[cb])
    os.system("date")
    print("-"*150)
    print("*"*150)
    print("-"*150)

"""
