import os, h5py
# Use scikit-learn to grid search the batch size and epochs
import numpy as np
import matplotlib.pyplot as plt

data_path = '/pylon5/as5phnp/tbilling/data/'
v1_no_w = data_path+'t21_snapshots_nowedge_v9.hdf5'
v2_w = data_path+'t21_snapshots_wedge_v9.hdf5'


fv1 = h5py.File(v1_no_w, 'r')
labelsv1 = np.asarray(fv1['Data'][u'snapshot_labels'][:,5])

fv2 = h5py.File(v2_w, 'r')
labelsv2 = np.asarray(fv2['Data'][u'snapshot_labels'][:,5])

plt.figure(figsize=(15,10))
# Normalize by setting density to true
#kwargs1 = dict(alpha=1.0, bins=100, density=None, stacked=True, histtype='step')
kwargs2 = dict(alpha=0.25, bins=100, density=None, stacked=True, histtype='stepfilled')
#kwargs3 = dict(alpha=1.0, bins=100, density=None, stacked=True, histtype='step')
kwargs4 = dict(alpha=1.0, bins=100, density=None, stacked=True, histtype='step')

# Plot
plt.hist(labelsv1, **kwargs2, color='lightcoral', label='Full Data')
#plt.hist(labelsv1[TrainIndex_7], **kwargs1, color='r', label='v7 Training')
plt.hist(labelsv2, **kwargs4, color='blue', label='Wedge Filtered Data')
#plt.hist(labelsv2[TrainIndex_8], **kwargs3, color='b', label='v8 Training')

plt.gca().set(title='', xlabel='Tau', ylabel='Count')
plt.xlim(0.035,0.085)
plt.ylim(0,22)
plt.legend()
plt.savefig("hist_v9_wedge_nowedge.png")
