import os, h5py, glob
# Use scikit-learn to grid search the batch size and epochs
import numpy as np
import matplotlib.pyplot as plt

n=0
factor=1000.

data_path = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/wedgecut/"
results = sorted(glob.glob(data_path+"results*.npy"))

fold = ["Fold 1","Fold 2","Fold 3","Fold 4","Fold 5","Fold 6","Fold 7","Fold 8","Fold 9","Fold 10"]

fig, axes = plt.subplots(figsize=(10,8),nrows=2, ncols=1, sharex=False, sharey=False)
for r, res in enumerate(results):
    #load reaults
    result=np.load(res)

    axes[0].scatter(result["truth"][:,n]/factor, result["prediction"][:,n]/factor,
    s=6, lw=0, alpha=0.9, label=fold[r])
    x = np.linspace(0.95*np.min(result["truth"][:,n]/factor),
                    1.05*np.max(result["truth"][:,n]/factor), 1000)
    axes[0].plot(x, x, 'k--',lw=5,alpha=0.2)


    axes[1].plot(x,(x-x)/x, 'k--',lw=5,alpha=0.2)
    relative_error = (result["prediction"][:,n]/factor-result["truth"][:,n]/factor)/result["truth"][:,n]
    axes[1].scatter(result["truth"][:,n]/factor, relative_error,s=6, lw=0, alpha=0.9, label=fold[r])

axes[1].set_xlabel("True")
axes[0].set_xlim(0.028,0.1)
axes[1].set_xlim(0.028,0.1)
axes[0].set_xscale('log')
axes[1].set_xscale('linear')

axes[0].set_ylabel("Predicted")
axes[0].set_ylim(0.028,0.1)
axes[1].set_ylabel(r"$\frac{Pred-True}{True}$")
axes[0].set_yscale('log')
axes[1].set_yscale('linear')

axes[0].legend(markerscale=2.5)
plt.savefig(data_path+"results_logScale.png")
#axes[1].legend(markerscale=2.5)
