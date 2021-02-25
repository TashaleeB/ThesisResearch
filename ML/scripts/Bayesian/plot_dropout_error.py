import numpy as np, matplotlib.pyplot as plt, gc, time, h5py

from matplotlib import gridspec

factor = 1000

training = False # False means dropout is not activated during dropout
wedge = True # Is the data wedge filtered
data_path = '/pylon5/as5phnp/tbilling/data/'

# Full Data
inputFile_nowedge = data_path+'t21_snapshots_nowedge_v12.hdf5'
result_nowedge = np.mean(np.load("nowedge/predictions.npz")["prediciton"][:,:,0]/factor, axis=0)
#yerr_nowedge = np.std(np.load("nowedge/predictions.npz")["prediciton"][:,:,0]/factor, axis=0)

# Wedge Filtered Data
inputFile_wedge = data_path+'t21_snapshots_wedge_v12.hdf5'
result_wedge = np.mean(np.load("wedge/predictions.npz")["prediciton"][:,:,0]/factor, axis=0)
#yerr_wedge = np.std(np.load("wedge/predictions.npz")["prediciton"][:,:,0]/factor, axis=0)

#outputdir = "/pylon5/as5phnp/tbilling/sandbox/bayesian/"

train_test_file = data_path + "train_test_index_80_20_split.npz"

def readLabels(inputFile, ind=None, **params):

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
  
def readImages(inputFile,ind, **params):
    # read in images onle

    print('reading data from', inputFile)

    f = h5py.File(inputFile, 'r')

    #if params['debug'] == True:
    #    data = np.asarray(f['Data'][u't21_snapshots'][ind,:,0:16,0:16])
    if 'crop' in params:
        print('cropping.')
        #use just the top corner of the images
        data = np.asarray(f['Data'][u't21_snapshots'][ind,:,0:params['crop'],0:params['crop']])
    else:
        #use everything!
        print('reading all data', len(ind))
        data = np.asarray(f['Data'][u't21_snapshots'][ind,:,:,:]) # (N_realizations, N_redshifts, N_pix, N_pix)
        print('loaded data', len(ind))

    print('finished loading data.', data.shape)

    data  = data.transpose(0,2,3,1) #(N_realizations, N_pix, N_pix, N_redshifts)

    return data, data[0].shape

def convert_to_true_tau(result, test_labels):
    # Convert to true tau units
    h_2 = 0.45321170409999995
    low_z_tau = 0.030029479627917934
    true_tau = low_z_tau + h_2 * result
    predicted_tau = low_z_tau + h_2 * test_labels/factor
    print(len(result))
    
    return true_tau,predicted_tau

# Load test Index Label
test_index = np.load(train_test_file)["test_index"]

# Load images and labels for test data
test_labels_nowedge = readLabels(inputFile=inputFile_nowedge,ind=None)[test_index,5]*factor
test_labels_nowedge = test_labels_nowedge.reshape(-1, 1)
#test_images_nowedge,input_shape = readImages(inputFile=inputFile_nowedge, ind=test_index)

test_labels_wedge = readLabels(inputFile=inputFile_wedge,ind=None)[test_index,5]*factor
test_labels_wedge = test_labels_wedge.reshape(-1, 1)
#test_images_wedge,input_shape = readImages(inputFile=inputFile_wedge, ind=test_index)

# convert to the true tau values
true_tau_nowedge,predicted_tau_nowedge = convert_to_true_tau(result=result_nowedge,
                                        test_labels=test_labels_nowedge)
true_tau_wedge,predicted_tau_wedge = convert_to_true_tau(result=result_wedge,
                                    test_labels=test_labels_wedge)

# calculate std of the true tau values
yerr_nowedge = np.std(true_tau_nowedge, axis=0)
yerr_wedge = np.std(true_tau_wedge, axis=0)

#fig, axes = plt.subplots(figsize=(10,8),nrows=2, ncols=1, sharex=False, sharey=False)
#plt.errorbar(true_tau,(true_tau-predicted_tau[:,0]), yerr=yerr, fmt='o', color="blue")
#plt.errorbar(true_tau, predicted_tau, xerr=xerr, yerr=yerr, fmt='-o')
#plt.scatter(true_tau, predicted_tau, s=6, lw=0, alpha=0.9, label=fold[r])
#x = np.linspace(0.95*np.min(true_tau), 1.05*np.max(true_tau), 1000)
#plt.plot(x,x-x, 'k--',lw=3,alpha=0.2)
#plt.xlabel(r"$\tau_\mathrm{true}$", fontsize=16)
#plt.ylabel(r"$\tau_\mathrm{true} - \tau_\mathrm{pred}$", fontsize=16)

figsize = (15, 10)
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(2, 1, hspace=0.0)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

# Plot data
ax1.errorbar(true_tau_nowedge,(true_tau_nowedge-predicted_tau_nowedge[:,0]), yerr=yerr_nowedge, fmt='o', color="blue")
ax2.errorbar(true_tau_wedge,(true_tau_wedge-predicted_tau_wedge[:,0]), yerr=yerr_wedge, fmt='o', color="blue")

x = np.linspace(0.95*np.min(true_tau_nowedge), 1.05*np.max(true_tau_nowedge), 1000)
ax1.plot(x,x-x, 'k--',lw=3,alpha=0.2)
x = np.linspace(0.95*np.min(true_tau_wedge), 1.05*np.max(true_tau_wedge), 1000)
ax2.plot(x,x-x, 'k--',lw=3,alpha=0.2)

# setup axes
ax1.text(0.0477, 0.0085, "Full 21 cm cube", size=16)
ax1.axhline(y=0, color="black", linestyle="--")
ax1.axhline(y=0.002, color="blue", linestyle="--", alpha=0.5, label="CMB EE cosmic variance")
ax1.axhline(y=-0.002, color="blue", linestyle="--", alpha=0.5)
ax2.axhline(y=0, color="black", linestyle="--")
ax2.axhline(y=0.002, color="blue", linestyle="--", alpha=0.5, label="CMB EE cosmic variance")
ax2.axhline(y=-0.002, color="blue", linestyle="--", alpha=0.5)

ax1.tick_params(labelcolor='k', labelsize='large', width=3)
ax1.set_xlim(0.047,0.072)
ax1.set_xscale('linear')

ax1.set_ylabel(r"$\tau_\mathrm{true} - \tau_\mathrm{pred}$", fontsize=16)
#ax1.set_ylim(0.048622955,0.07100295)
ax1.set_yscale('linear')

ax2.text(0.0477, 0.0085, "Wedge cut 21 cm cube", size=16)
ax2.tick_params(labelcolor='k', labelsize='large', width=3)
ax2.set_xlabel(r"$\tau_\mathrm{true}$", fontsize=16)
ax2.set_xlim(0.047,0.072)
ax2.set_xscale('linear')

ax2.set_ylabel(r"$\tau_\mathrm{true} - \tau_\mathrm{pred}$", fontsize=16)
ax2.set_yscale('linear')


plt.savefig("predictions_diff_.png")
