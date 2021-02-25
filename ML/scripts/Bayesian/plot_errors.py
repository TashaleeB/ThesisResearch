import glob, h5py
import numpy as np
from scipy.stats import binned_statistic

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

fontsize = 16


data_path = '/pylon5/as5phnp/tbilling/data/'


    
outputdir = "/pylon5/as5phnp/tbilling/sandbox/bayesian/"

train_test_file = data_path + "train_test_index_80_20_split.npz"

n=0
batch_size = 32
N_EPOCH = 200
factor =1000.


def readLabels(ind=None, **params):

    # read in labels only

    f = h5py.File(inputFile, 'r')

    if ind is None:
        labels = np.asarray(f['Data'][u'snapshot_labels'])  #(N_realizations, N_parameters)
    else:
        labels = np.asarray(f['Data'][u'snapshot_labels'][:, ind])

    if labels.ndim == 1:
        print('training on just one param.')
        print('starting with the following shape, dim:', labels.shape, labels.ndim)
        if labels.ndim > 1:
            labels = labels[:, params['predictoneparam']]
    elif labels.shape[1] == 2:
        print('training on two params.')
        print('starting with the following shape, dim:', labels.shape, labels.ndim)
        if labels.ndim > 1:
            labels = labels[:, ind]

    #if there's only one label per image, we'll have to reshape it:
    if labels.ndim == 1:
        print('reshaping data...')
        labels = labels.reshape(-1, 1)

    return labels

# Load Index Label
test_index = np.load(train_test_file)["test_index"]


def pseudotau2tau(ptau):
    tau = ptau.copy()
    tau /= 1000.0  # Tasha's scaling
    # cosmology
    h = 0.67321
    low_z_tau = 0.03
    tau = low_z_tau + h**2 * tau
    return tau


def bin_center(bin_edges):
    return bin_edges[:-1] + np.diff(bin_edges) / 2



# read in data and compute statistics
files_modes_removed = sorted(glob.glob("/pylon5/as5phnp/tbilling/sandbox/bayesian/dropout/wedge/predictions.npz"))
files_no_modes_removed = sorted(glob.glob("/pylon5/as5phnp/tbilling/sandbox/bayesian/dropout/nowedge/predictions.npz"))

full_prediction = pseudotau2tau(np.load(files_no_modes_removed[0])["prediciton"][:,:,0])
cut_prediction = pseudotau2tau(np.load(files_modes_removed[0])["prediciton"][:,:,0])
 
for w in [True, False]:
    if wedge == False:
        inputFile = data_path+'t21_snapshots_nowedge_v12.hdf5'
        labels = np.asarray(inputFile['Data'][u'snapshot_labels'][:])[test_index,5]*factor
        full_truth = labels.reshape(-1, 1)

    if wedge == True:
        inputFile = data_path+'t21_snapshots_wedge_v12.hdf5'
        labels = np.asarray(inputFile['Data'][u'snapshot_labels'][:])[test_index,5]*factor
        cut_truth = labels.reshape(-1, 1)





    err_full = full["truth"] - full["prediction"]
    #frac_err_full = (full["truth"] - full["prediction"]) / full["truth"]

    bias_full, bias_full_bin_edges, _ = binned_statistic(
        full["truth"].squeeze(), err_full.squeeze(), statistic="mean"
    )
    std_full, std_full_bin_edges, _ = binned_statistic(
        full["truth"].squeeze(), err_full.squeeze(), statistic="std"
    )

    err_cut = cut["truth"] - cut["prediction"]
    #frac_err_cut = (cut["truth"] - cut["prediction"]) / cut["truth"]

    bias_cut, bias_cut_bin_edges, _ = binned_statistic(
        cut["truth"].squeeze(), err_cut.squeeze(), statistic="mean"
    )
    std_cut, std_cut_bin_edges, _ = binned_statistic(
        cut["truth"].squeeze(), err_cut.squeeze(), statistic="std"
    )

    # make a figure
    matplotlib.rc("text", usetex=True)
    matplotlib.rc("font", family="serif")
    matplotlib.rc("lines", linewidth=2)

    figsize = (7, 7)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, hspace=0.0)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    # ax1 = fig.add_subplot(211)
    # ax2 = fig.add_subplot(212)

    ax1.axhline(y=0, color="black", linestyle="--")
    ax1.axhline(
        y=0.002, color="blue", linestyle="--", alpha=0.5, label="CMB EE cosmic variance"
    )
    ax1.axhline(y=-0.002, color="blue", linestyle="--", alpha=0.5)
    ax2.axhline(y=0, color="black", linestyle="--")
    ax2.axhline(
        y=0.002, color="blue", linestyle="--", alpha=0.5, label="CMB EE cosmic variance"
    )
    ax2.axhline(y=-0.002, color="blue", linestyle="--", alpha=0.5)

    ax1.plot(full["truth"], err_full, ".", color="gray", alpha=0.5)
    ax1.errorbar(
        bin_center(bias_full_bin_edges),
        bias_full,
        yerr=std_full,
        # capsize=3,
        color="red",
        marker="o",
        label="Mean bias and variance",
    )
    # ax1.set_xlabel(r'$\tau_\mathrm{true}$', size=fontsize)
    ax1.set_ylabel(r"$\tau_\mathrm{true} - \tau_\mathrm{pred}$", size=fontsize)
    ax1.text(0.0477, 0.003, "Full 21 cm cube", size=fontsize)
    leg = ax1.legend(prop={"size": fontsize})

    ax2.plot(cut["truth"], err_cut, ".", color="gray", alpha=0.5)
    ax2.errorbar(
        bin_center(bias_cut_bin_edges),
        bias_cut,
        yerr=std_cut,
        # capsize=3,
        color="red",
        marker="o",
        label="Mean bias and variance",
    )

    ax1.set_ylim([-0.004, 0.004])
    ax2.set_ylim([-0.004, 0.004])
    ax2.set_yticks([-0.004, -0.002, 0, 0.002, 0.004])
    yticks = ax2.get_yticks()
    ax2.set_yticks(yticks[:-1])
    ax2.set_xlim([0.0475, 0.0675])
    ax1.tick_params(labelbottom=False)
    ax1.tick_params(axis="both", labelsize=fontsize)
    ax2.tick_params(axis="both", labelsize=fontsize)

    ax2.set_xlabel(r"$\tau_\mathrm{true}$", size=fontsize)
    ax2.set_ylabel(r"$\tau_\mathrm{true} - \tau_\mathrm{pred}$", size=fontsize)
    ax2.text(0.0477, 0.003, "Wedge cut 21 cm cube", size=fontsize)
    output = "bias_variance.pdf"
    print(f"Saving {output}...")
    fig.savefig(output, bbox_inches="tight")

    return


if __name__ == "__main__":
    plot_errors()





   ...: factor = 1000
   ...:
   ...: training = False # False means dropout is not activated during dropout
   ...: wedge = False # Is the data wedge filtered
   ...: data_path = '/pylon5/as5phnp/tbilling/data/'
   ...:
   ...: if wedge == False:
   ...:     inputFile = data_path+'t21_snapshots_nowedge_v12.hdf5'
   ...:
   ...: if wedge == True:
   ...:     inputFile = data_path+'t21_snapshots_wedge_v12.hdf5'
   ...:
   ...: #outputdir = "/pylon5/as5phnp/tbilling/sandbox/bayesian/"
   ...:
   ...: train_test_file = data_path + "train_test_index_80_20_split.npz"
   ...:
   ...: def readLabels(ind=None, **params):
   ...:
   ...:     # read in labels only
   ...:
   ...:     f = h5py.File(inputFile, 'r')
   ...:
   ...:     if ind is None:
   ...:         labels = np.asarray(f['Data'][u'snapshot_labels'])  #(N_realizations, N_parameters)
   ...:     else:
   ...:         labels = np.asarray(f['Data'][u'snapshot_labels'][:, ind])
   ...:
   ...:     if labels.ndim == 1:
   ...:         print('training on just one param.')
   ...:         print('starting with the following shape, dim:', labels.shape, labels.ndim)
   ...:         if labels.ndim > 1:
   ...:             labels = labels[:, params['predictoneparam']]
   ...:     elif labels.shape[1] == 2:
   ...:         print('training on two params.')
   ...:         print('starting with the following shape, dim:', labels.shape, labels.ndim)
   ...:         if labels.ndim > 1:
   ...:             labels = labels[:, ind]
   ...:
   ...:     #if there's only one label per image, we'll have to reshape it:
   ...:     if labels.ndim == 1:
   ...:         print('reshaping data...')
   ...:         labels = labels.reshape(-1, 1)
   ...:
   ...:     return labels


   ...: def readImages(ind, **params):
   ...:     # read in images onle
   ...:
   ...:     print('reading data from', inputFile)
   ...:
   ...:     f = h5py.File(inputFile, 'r')
   ...:
   ...:     #if params['debug'] == True:
   ...:     #    data = np.asarray(f['Data'][u't21_snapshots'][ind,:,0:16,0:16])
   ...:     if 'crop' in params:
   ...:         print('cropping.')
   ...:         #use just the top corner of the images
   ...:         data = np.asarray(f['Data'][u't21_snapshots'][ind,:,0:params['crop'],0:params['crop']])
   ...:     else:
   ...:         #use everything!
   ...:         print('reading all data', len(ind))
   ...:         data = np.asarray(f['Data'][u't21_snapshots'][ind,:,:,:]) # (N_realizations, N_redshifts, N_pix, N_pix)
   ...:         print('loaded data', len(ind))
   ...:
   ...:     print('finished loading data.', data.shape)
   ...:
   ...:     data  = data.transpose(0,2,3,1) #(N_realizations, N_pix, N_pix, N_redshifts)
   ...:
   ...:     return data, data[0].shape
   ...:
   ...:
   ...: # Load Index Label
   ...: test_index = np.load(train_test_file)["test_index"]
   ...:
   ...: # Load images and labels for training and testing
   ...:
   ...: test_labels = readLabels(ind=None)[test_index,5]*factor
   ...: test_labels = test_labels.reshape(-1, 1)
   ...: test_images,input_shape = readImages(ind=test_index)
   ...:
   ...: factor = 1000
   ...: result = np.mean(np.load("nowedge/predictions.npz")["prediciton"][:,:,0]/factor, axis=0)
   ...: yerr = np.std(np.load("nowedge/predictions.npz")["prediciton"][:,:,0]/factor, axis=0)
   ...:
   ...: # Convert to true tau units
   ...: h_2 = 0.45321170409999995
   ...: low_z_tau = 0.030029479627917934
   ...: true_tau = low_z_tau + h_2 * result
   ...: predicted_tau = low_z_tau + h_2 * test_labels/factor
   ...: print(len(result))
   ...:
   ...: plt.figure(figsize=(10,10))
   ...: plt.errorbar(true_tau,(predicted_tau[:,0]-true_tau), yerr=yerr, fmt='o')
   ...: #plt.errorbar(true_tau, predicted_tau, xerr=xerr, yerr=yerr, fmt='-o')
   ...: #plt.scatter(true_tau, predicted_tau, s=6, lw=0, alpha=0.9, label=fold[r])
   ...: x = np.linspace(0.95*np.min(true_tau), 1.05*np.max(true_tau), 1000)
   ...: plt.plot(x,x-x, 'k--',lw=3,alpha=0.2)
    ...: plt.xlabel(r"$\tau_\mathrm{true}$", fontsize=16)
    ...: plt.ylabel(r"$\tau_\mathrm{true} - \tau_\mathrm{pred}$", fontsize=16)
   ...: plt.savefig("nowedge/predictions_diff.png")
