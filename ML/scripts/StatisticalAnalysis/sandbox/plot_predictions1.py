import os, h5py, glob
# Use scikit-learn to grid search the batch size and epochs
import numpy as np
import matplotlib.pyplot as plt

n=0
factor=1000.

data_path = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/wedgecut/"
results = sorted(glob.glob(data_path+"results*.npy"))

fold = ["Fold 1","Fold 2","Fold 3","Fold 4","Fold 5","Fold 6","Fold 7","Fold 8","Fold 9"]

plt.figure(figsize=(10,8))
for r, res in enumerate(results):
    #load reaults
    result=np.load(res)

    plt.scatter(result["truth"][:,n]/factor, result["prediction"][:,n]/factor,
                s=6, lw=0, alpha=0.9, label=fold[r])
    x = np.linspace(0.95*np.min(result["truth"][:,n]/factor),
                1.05*np.max(result["truth"][:,n]/factor), 1000)
    plt.plot(x, x, 'k--',lw=5,alpha=0.2)

plt.xlabel('True')
plt.xlim(0.028,0.1)
#plt.xscale('log')

plt.ylabel('Predicted')
axes[0].set_ylim(0.028,0.1)
#plt.yscale('log')
plt.legend(markerscale=2.5)

filename = outputdir+'PredvsTruth{}_{:d}.png'.format(steps,fold)
plt.savefig(filename)
plt.clf()
