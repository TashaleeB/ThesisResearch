from __future__ import print_function, division, absolute_import
import os, h5py, glob
# Use scikit-learn to grid search the batch size and epochs
import numpy as np
import matplotlib.pyplot as plt

n=0
factor=1000.
h_2 = 0.45321170409999995
low_z_tau = 0.030029479627917934

data_path = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/opt_versions/wedgefilter_v9/"

results = glob.glob(data_path+"bestmodel_pred_results_*.npy")
fold = ["Fold 1","Fold 2","Fold 3","Fold 4","Fold 5","Fold 6","Fold 7","Fold 8","Fold 9","Fold 10"]

fig, axes = plt.subplots(figsize=(10,8),nrows=2, ncols=1, sharex=False, sharey=False)
for r, res in enumerate(results):
    #load reaults
    result=np.load(res)
    
    # Convert to true tau units
    true_tau = low_z_tau + h_2 * result["truth"][:,n]/factor
    predicted_tau = low_z_tau + h_2 * result["prediction"][:,n]/factor

    axes[0].scatter(true_tau, predicted_tau, s=6, lw=0, alpha=0.9, label=fold[r])
    x = np.linspace(0.95*np.min(true_tau), 1.05*np.max(true_tau), 1000)
    axes[0].plot(x, x, 'k--',lw=1,alpha=0.2)

    axes[1].plot(x-x, 'k--',lw=5,alpha=0.2)
    relative_error = (predicted_tau-true_tau)/true_tau
    axes[1].scatter(true_tau, relative_error,s=6, lw=0, alpha=0.9, label=fold[r])

axes[1].set_xlabel('True',fontsize=16)
axes[0].set_xlim(0.028,0.1)
axes[1].set_xlim(0.028,0.1)
axes[0].set_xscale('log')
axes[1].set_xscale('linear')

axes[0].set_ylabel('Predicted',fontsize=16)
axes[0].set_ylim(0.028,0.1)
axes[1].set_ylabel('Residual Relative Difference',fontsize=16)
axes[0].set_yscale('log')
axes[1].set_yscale('linear')

axes[0].tick_params(labelcolor='k', labelsize='large', width=3)
axes[1].tick_params(labelcolor='k', labelsize='large', width=3)

#axes[0].legend(markerscale=2.5)
plt.savefig(data_path+"bestmodel_residual_linearScale.png")
#axes[1].legend(markerscale=2.5)
