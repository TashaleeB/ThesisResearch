from __future__ import print_function, division, absolute_import
import os, h5py, glob
# Use scikit-learn to grid search the batch size and epochs
import numpy as np
import matplotlib.pyplot as plt

n=0
factor=1000.

data_path = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/"
histories = sorted(glob.glob(data_path+"history_loss_*.npz"))
#history_name = np.array(["loss", "mae", "mape", "mse", "val_loss", "val_mae", "val_mape", "val_mse"])

for i in np.arange(len(histories)):
    history = sorted(glob.glob(data_path+"history*{}.npz".format(str(i))))
    
    loss = np.load(history[0])['metric']
    val_loss = np.load(history[1])['metric']
    
    plt.figure(figsize=(15,10))
    plt.plot(loss[1:],".-")
    plt.plot(val_loss[1:],".-")

    plt.legend(['Train', 'Test'], loc='upper right', markerscale=2.5)
    plt.savefig(data_path+'Loss_Epoch_linearity_{:d}.png'.format(i))
    plt.clf()
