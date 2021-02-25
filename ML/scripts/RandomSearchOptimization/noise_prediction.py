#printf "\e[?2004l"
# needs to be ran in hp_opt environment with Tensorflow version
# tf.__version__ : '2.0.0'
# anything related to optimization you should use this version of tf.

from __future__ import print_function, division, absolute_import

# Imports
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import h5py, glob, os
import numpy as np, matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K


import gc
gc.enable()


factor =1000.
istart = 0
def Mean_Squared_over_true_Error(y_true, y_pred):
    # Create a custom loss function that divides the difference by the true
    #if not K.is_tensor(y_pred):
    if not K.is_keras_tensor(y_pred):
        y_pred = K.constant(y_pred)

    y_true = K.cast(y_true, y_pred.dtype)
    diff_ratio = K.square((y_pred - y_true)/K.clip(K.abs(y_true),K.epsilon(),None))
    loss = K.mean(diff_ratio, axis=-1)
    # Return a function

    return loss


noise_data = sorted(glob.glob("/pylon5/as5phnp/tbilling/data/*_noise_*.h5"))
models = ["no_modes_removed/CNN_model_nowedge_3.h5", "modes_removed/CNN_model_wedge_3.h5"]
nowedge_noise = noise_data[:3] + noise_data[6:9]
wedge_noise = noise_data[3:3+3] + noise_data[6+3:]

def savePreds(model, eval_data, eval_labels, Ntot, fold, filename):
    outputFile = "/pylon5/as5phnp/tbilling/sandbox/redo_mlpaper/eval_pred_results_"+filename+"npy"

    # The Predict() method -  is for the actual prediction. It generates output predictions for the input samples.
    preds = model.predict(eval_data, verbose=0).flatten() #0 = silent

    Nregressparams = len(eval_labels[0])

    results = np.zeros((Ntot, Nregressparams),
                           dtype = [('truth', 'f'), ('prediction', 'f'), ('fold', 'i')])
    iend = istart+len(eval_labels)

    #print('istart and iend', istart, iend)

    results['fold'][istart:iend] = fold
    results['truth'][istart:iend] = eval_labels
    #results['truth'][istart-100:iend-100] = eval_labels
    for n in range(Nregressparams):
        results['prediction'][istart:iend,n] = preds[n::Nregressparams]
        #results['prediction'][istart-100:iend-100,n] = preds[n::Nregressparams]

    np.save(outputFile, results)
#del model
#gc.collect()
#K.clear_session()
for m in len(models):
    

model = load_model(models[m],custom_objects= {"Mean_Squared_over_true_Error":Mean_Squared_over_true_Error}, compile=False)
model.compile(optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.),
    loss=Mean_Squared_over_true_Error,
    metrics=['mse', 'mae', 'mape'])
for nn, noise in enumerate([nowedge_noise,wedge_noise][m]):
    print('saving predictions ... {}'.format(noise))

    if nn == 3 or nn > 3:
        testlabels = np.array(h5py.File(noise, 'r')['test_labels'])*factor
        testimages = h5py.File(noise, 'r')['test_images']
        #trainlabels = h5py.File(noise_data[-1], 'r')['train_labels'][0]*factor
    else:
        testlabels = h5py.File(noise, 'r')['test_labels']
        testimages = h5py.File(noise, 'r')['test_images']
        #trainlabels = h5py.File(noise_data[-1], 'r')['train_labels'][0]

    savePreds(model, testimages, testlabels, len(testlabels), fold=0, filename=os.path.basename(noise)[:-2])#, istart=0, outdir=outputdir)

factor =1000.
n=0
h_2 = 0.45321170409999995
low_z_tau = 0.030029479627917934

results = sorted(glob.glob("eval_pred_results_*noise*.npy"))

names = ["Full: 0.001", "Full: 0.01", "Full: 0.1", "Wedge Cut: 0.001", "Wedge Cut: 0.01", "Wedge Cut: 0.1", "Full: 0.001", "Full: 0.01", "Full: 0.1", "Wedge Cut: 0.001", "Wedge Cut: 0.01", "Wedge Cut: 0.1"]

results = sorted(glob.glob("eval_pred_results_*noise*.npy"))
fig, axes = plt.subplots(figsize=(15,10),nrows=2, ncols=2, sharex=False, sharey=False)
for r, res in enumerate(results):
    #load reaults
    if r > 5:
        result=np.load(res)

        # Convert to true tau units
        true_tau = low_z_tau + h_2 * result["truth"][:,n]/factor
        predicted_tau = low_z_tau + h_2 * result["prediction"][:,n]/factor

        #axes[0,1].plot(true_tau, predicted_tau, "d", label=names[r])
        axes[0,1].scatter(true_tau, predicted_tau, s=20, lw=0, alpha=0.9, label=names[r])#os.path.basename(res)[:-3])
        x = np.linspace(0.95*np.min(true_tau), 1.05*np.max(true_tau), 1000)
        axes[0,1].plot(x, x, 'k--',lw=2,alpha=0.2)


        axes[1,1].plot(x-x, 'k--',lw=2,alpha=0.2)
        relative_error = (predicted_tau-true_tau)/true_tau
        #axes[1,1].plot(true_tau, relative_error, "d", label=names[r])
        axes[1,1].scatter(true_tau, relative_error,s=20, lw=0, alpha=0.9)#, label=fold[r])

    else:
        result=np.load(res)

        # Convert to true tau units
        true_tau = low_z_tau + h_2 * result["truth"][:,n]/factor
        predicted_tau = low_z_tau + h_2 * result["prediction"][:,n]/factor

        #axes[0,0].plot(true_tau, predicted_tau, "*", label=names[r])
        axes[0,0].scatter(true_tau, predicted_tau, s=20, lw=0, alpha=0.9, label=names[r])#os.path.basename(res)[:-3])
        x = np.linspace(0.95*np.min(true_tau), 1.05*np.max(true_tau), 1000)
        axes[0,0].plot(x, x, 'k--',lw=2,alpha=0.2)


        axes[1,0].plot(x-x, 'k--',lw=2,alpha=0.2)
        relative_error = (predicted_tau-true_tau)/true_tau
        #axes[1,0].plot(true_tau, relative_error, ".", label=names[r])
        axes[1,0].scatter(true_tau, relative_error,s=20, lw=0, alpha=0.9)

axes[1,0].set_xlabel(r"$\tau_{True}$", fontsize=16)
axes[1,1].set_xlabel(r"$\tau_{True}$", fontsize=16)
axes[0,0].set_xlim(0.0475,0.0675)
axes[0,1].set_xlim(0.0475,0.0675)
axes[1,0].set_xlim(0.0475,0.0675)
axes[1,1].set_xlim(0.0475,0.0675)
axes[0,0].set_xscale('linear')
axes[1,0].set_xscale('linear')

axes[0,0].set_ylabel("Predicted", fontsize=16)
axes[0,0].set_ylim(0.0355,0.0675)
axes[0,1].set_ylim(0.0355,0.0675)
axes[1,0].set_ylabel("Residual Relative Difference", fontsize=16)
axes[0,0].set_yscale('linear')
axes[1,0].set_yscale('linear')
axes[0,0].text(0.06, 0.045, "Planck18", size=16)
axes[0,1].text(0.06, 0.045, "WMAP9", size=16)
axes[0,0].legend(markerscale=1.0)
axes[0,1].legend(markerscale=1.0)
axes[0,0].tick_params(labelcolor='k', labelsize='large', width=3)
axes[0,1].tick_params(labelcolor='k', labelsize='large', width=3)
axes[1,0].tick_params(labelcolor='k', labelsize='large', width=3)
axes[1,1].tick_params(labelcolor='k', labelsize='large', width=3)
plt.tight_layout()
plt.savefig("bestmodel_noise_2.png")
    

results = sorted(glob.glob("eval_pred_results_t21_snapshots_*.npy"))
fold = ["Full", "Wedge Cut"]
plt.figure(figsize=(8,8))
for r, res in enumerate(results):
    #load reaults
    result=np.load(res)
    true_tau = low_z_tau + h_2 * result["truth"][:,n]/factor
    predicted_tau = low_z_tau + h_2 * result["prediction"][:,n]/factor

    plt.scatter(true_tau, predicted_tau, s=20, lw=0, alpha=0.9, label=fold[r])
    x = np.linspace(0.95*np.min(true_tau), 1.05*np.max(true_tau), 1000)
    plt.plot(x, x, 'k--',lw=2,alpha=0.2)

plt.xlabel(r"$\tau_{True}$", fontsize=16)
plt.xlim(0.0475,0.0675)


plt.ylabel('Predicted', fontsize=16)
plt.ylim(0.0475,0.0675)
plt.tick_params(labelcolor='k', labelsize='large', width=3)
plt.legend(markerscale=1.5)
plt.savefig("bestmodel_residual_v13.png")
