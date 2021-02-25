from __future__ import print_function, division, absolute_import
import numpy as np, matplotlib.pyplot as plt, os, h5py, glob

history = np.load("10-3.10-3.10-3/denseflipout_CNN_nowedge_history.npz")

val_loss = history['val_loss']
#loss = history['loss']

difference_list = []
for i in np.arange(len(val_loss)-1):
    difference = val_loss[i+1] - val_loss[i]
    difference_list.append(difference)

# Difference Plots
difference_array = np.array(difference_list)

print("Making plot")
plt.figure(figsize=(15,10))
plt.plot(difference_array[0::16], "ro", label= "Validation")
#plt.plot(difference_array[0::16], "ro", label= "Training")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.set_yscale('log')
#plt.ylim(-500,500)
plt.legend(markerscale=2.5)
plt.savefig("10-3.10-3.10-3/difference_validation_loss_epoch.png")
#plt.savefig("10-3.10-3.10-3/difference_training_loss_epoch.png")

# Combined Training and Validation Plots
plt.figure(figsize=(15,10))
plt.plot(history['loss'][0::16], "r-", label= "Training")
plt.plot(history['val_loss'][0::16], "b-", label= "Validation")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.set_yscale('log')
#plt.ylim(-500,500)
plt.legend(markerscale=2.5)
plt.savefig("10-3.10-3.10-3/training_validation_loss_epoch.png")

# Training Plots
plt.figure(figsize=(15,10))
plt.plot(history['loss'][0::16], "r-", label= "Training")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.set_yscale('log')
#plt.ylim(-500,500)
plt.legend(markerscale=2.5)
plt.savefig("10-3.10-3.10-3/training_loss_epoch.png")

# Validation Plots
plt.figure(figsize=(15,10))
plt.plot(history['val_loss'][0::16], "b-", label= "Validation")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.set_yscale('log')
#plt.ylim(-500,500)
plt.legend(markerscale=2.5)
plt.savefig("10-3.10-3.10-3/validation_loss_epoch.png")
