"""
The purpose of this script is to read in all of the eval_modelcomplexity.npz files and find the maan loss of each model and the the variance.
"""

from __future__ import print_function, division, absolute_import

# Imports
import os, sys, h5py, glob
import numpy as np, matplotlib.pyplot as plt

path = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/opt_versions/"
output = path+"wedgefilter_v9/"
inputFile = output+"eval_score_*data.npz"


# Load in all npz files and sort the list from 1 to 15 (then end)
eval_scores_names = []

file_names = sorted(glob.glob(inputFile))
for s in range(len(file_names)):
    eval_scores_names.append(output+'eval_score_{}data.npz'.format(s+1))

eval_scores= np.array(eval_scores_names)
del(file_names)

scores = []
param_count = []

for e in eval_scores_names:
    data = np.load(e)
    scores.append(data['scores'])
    param_count.append(data['param_count'])
scores = np.array(scores) # [fold number, models tested]
param_count = np.array(param_count)

# sort them with respect to the number of parameters (small to large)
sorted_param_count = np.array(param_count)[:,np.array(param_count[0,:]).argsort()][0,:]
sorted_scores = np.array(scores)[np.array(param_count[0,:]).argsort()]

# calculate the statistics of each model
mean_sorted_scores = np.mean(sorted_scores, axis=0)
std_sorted_scores = np.std(sorted_scores, axis=0)
var_sorted_scores = np.var(sorted_scores, axis=0)

# the st
plt.figure(figsize=(15,15))
for i, txt in enumerate(mean_sorted_scores):
     plt.plot(sorted_param_count[i], mean_sorted_scores[i], 'r*')
     plt.annotate("Mean: "+str("%.3f" % mean_sorted_scores[i]), xy=(sorted_param_count[i], mean_sorted_scores[i]))

     plt.plot(sorted_param_count[i], std_sorted_scores[i], 'bo')
     plt.annotate("STD: "+str("%.3f" % mean_sorted_scores[i]), xy=(sorted_param_count[i], std_sorted_scores[i]))

     plt.plot(sorted_param_count[i], var_sorted_scores[i], 'b*')
     plt.annotate("Variance: "+str("%.3f" % mean_sorted_scores[i]), xy=(sorted_param_count[i], var_sorted_scores[i]))
     plt.annotate("Param, Var: "+str(sorted_param_count[i])+", "+str("%.3f" % mean_sorted_scores[i]), xy=(sorted_param_count[i], var_sorted_scores[i]))
     #plt.text(x+.03, y+.03, word, fontsize=9)

plt.ylabel("Statistics")
plt.xlabel("Number of Trainable Parameters")
plt.legend(markerscale =2)
plt.show()
