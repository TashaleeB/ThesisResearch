from __future__ import print_function, division, absolute_import
import os, h5py, glob, io
# Use scikit-learn to grid search the batch size and epochs
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

n=0
factor=1000.
h_2 = 0.45321170409999995
low_z_tau = 0.030029479627917934
coefficients = []

data_path = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/opt_versions/200steps_v9/" # No wedge filtering
#data_path = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/opt_versions/wedgefilter_v9/"# Wedge Filterd

results = glob.glob(data_path+"bestmodel_pred_results_*.npy")
fold = ["Fold 1","Fold 2","Fold 3","Fold 4","Fold 5","Fold 6","Fold 7","Fold 8","Fold 9","Fold 10"]


def read_results(filename):

    #read in file
    result=np.load(filename)
    truth = (result["truth"][:,n]/factor)[:5*20]
    prediction = (result["prediction"][:,n]/factor)[:5*20]

    return [truth, prediction]


def statistics_plot_dev(coefficient):
        coefficients = np.array(coefficient,dtype=np.float64)
    
        mean_value = np.mean(coefficients,axis=0)
        mean = np.zeros_like(coefficients)
        mean[:,0]=mean_value[0]
        mean[:,1]=mean_value[1]
    
        return mean, coefficients

# -----------------------------------------------
# No wedge filtering
# -----------------------------------------------
u = u"""Intercept,Slope
5.421855747068244e-05, 0.9924262447700102
-0.0002882792936294565, 1.0000046227139294
-0.000257784479148801, 0.999734020898306
-0.0008852199621917564, 1.011030073293924
-0.00045167164309282276, 1.0013431233736403
0.0006507348475776331, 0.9815821599357388
-0.000720321508859402, 1.0070076399283203
4.793412680076381e-05, 0.9941265751794229
-0.00010379406667131164, 0.9963022374917446
-0.0004790403162922771, 1.0011809886396998"""
bb = np.array([[5.421855747068244e-05, 0.9924262447700102],
[-0.0002882792936294565, 1.0000046227139294],
[-0.000257784479148801, 0.999734020898306],
[-0.0008852199621917564, 1.011030073293924],
[-0.00045167164309282276, 1.0013431233736403],
[0.0006507348475776331, 0.9815821599357388],
[-0.000720321508859402, 1.0070076399283203],
[4.793412680076381e-05, 0.9941265751794229],
[-0.00010379406667131164, 0.9963022374917446],
[-0.0004790403162922771, 1.0011809886396998]])

mean_, coeff =statistics_plot_dev(bb)

# -----------------------------------------------
# Wedge Filterd
# -----------------------------------------------
u = u"""Intercept,Slope
    1.37832505e-03,9.73985875e-01
    9.74338875e-04,9.80718787e-01
    7.35699333e-04,9.84541122e-01
    -7.31605055e-04,1.01211542e+00
    1.07818650e-03,9.80426201e-01
    1.02408265e-03,9.80178626e-01
    8.12751298e-04,9.90751611e-01
    1.09524624e-03,9.79628791e-01
    -3.45495928e-04,1.00534564e+00
    -1.81619994e-04,1.00013448e+00"""
bb = np.array([[ 1.37832505e-03,  9.73985875e-01],
[ 9.74338875e-04,  9.80718787e-01],
[ 7.35699333e-04,  9.84541122e-01],
[-7.31605055e-04,  1.01211542e+00],
[ 1.07818650e-03,  9.80426201e-01],
[ 1.02408265e-03,  9.80178626e-01],
[ 8.12751298e-04,  9.90751611e-01],
[ 1.09524624e-03,  9.79628791e-01],
[-3.45495928e-04,  1.00534564e+00],
[-1.81619994e-04,  1.00013448e+00]])
mean_, coeff =statistics_plot_dev(bb)
# -----------------------------------------------

data = io.StringIO(u)

df = pd.read_csv(data, sep=",", index_col=0)
#plt.clf()
#plt.boxplot(df["Slope"])
#plt.boxplot(df["Intercept"])
#plt.show()

fig, axes = plt.subplots(figsize=(15,10),nrows=2, ncols=2, sharex=False, sharey=False)
for r, res in enumerate(results):
    #load reaults
    result=np.load(res)
    
    # Convert to true tau units
    true_tau = low_z_tau + h_2 * result["truth"][:,n]/factor
    predicted_tau = low_z_tau + h_2 * result["prediction"][:,n]/factor

    axes[0,0].scatter(true_tau, predicted_tau, s=6, lw=0, alpha=0.9, label=fold[r])
    x = np.linspace(0.95*np.min(true_tau), 1.05*np.max(true_tau), 1000)
    axes[0,0].plot(x, x, 'k--',lw=1,alpha=0.2)


    axes[1,0].plot(x-x, 'k--',lw=1,alpha=0.2)
    relative_error = (predicted_tau-true_tau)/true_tau
    axes[1,0].scatter(true_tau, relative_error,s=6, lw=0, alpha=0.9, label=fold[r])

# plot box plot
axes[0,1].boxplot(df["Slope"], labels=" ")
axes[0,1].set_ylabel("")
axes[0,1].set_xlabel("Best Fit Model", fontsize=18)

# Plot distance from mean
axes[1,1].plot(np.arange(1,11),coeff,'o')
axes[1,1].plot(np.arange(1,11),mean_,'k--',lw=1,markersize=4)
axes[1,1].set_xlabel("Fold Number", fontsize=18)
axes[1,1].set_ylabel("Distance from Mean", fontsize=18)

axes[1,0].set_xlabel("True", fontsize=18)
axes[0,0].set_xlim(0.0475,0.0675)
axes[1,0].set_xlim(0.0475,0.0675)
axes[0,0].set_xscale('linear')
axes[1,0].set_xscale('linear')

axes[0,0].set_ylabel("Predicted", fontsize=18)
axes[0,0].set_ylim(0.028,0.1)
axes[1,0].set_ylabel("Residual Relative Difference", fontsize=18)
axes[0,0].set_yscale('linear')
axes[1,0].set_yscale('linear')
 
axes[0,0].legend(markerscale=2.5)
plt.savefig(data_path+"bestmodel_4residual.png")


"""
bb = array([[ 1.37832505e-03,  9.73985875e-01],
           [ 9.74338875e-04,  9.80718787e-01],
           [ 7.35699333e-04,  9.84541122e-01],
           [-7.31605055e-04,  1.01211542e+00],
           [ 1.07818650e-03,  9.80426201e-01],
           [ 1.02408265e-03,  9.80178626e-01],
           [ 8.12751298e-04,  9.90751611e-01],
           [ 1.09524624e-03,  9.79628791e-01],
           [-3.45495928e-04,  1.00534564e+00],
           [-1.81619994e-04,  1.00013448e+00]])
"""
