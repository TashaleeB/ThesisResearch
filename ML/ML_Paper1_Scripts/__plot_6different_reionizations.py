from __future__ import print_function, division, absolute_import
import os, h5py, glob
# Use scikit-learn to grid search the batch size and epochs
import numpy as np
import matplotlib.pyplot as plt

n=0
factor=1000.

data_path = "/pylon5/as5phnp/tbilling/data/"

data_name1 = "t21_snapshots_wedge_v9.hdf5"
inputFile1 = data_path+data_name1

data_name2 = "t21_snapshots_nowedge_v9.hdf5"
inputFile2 = data_path+data_name2

def readImages(inputFile, ind, **params):

    print('reading data from', inputFile)

    f = h5py.File(inputFile, 'r')

    if 'crop' in params:
        #use just the top corner of the images
        data = np.asarray(f['Data'][u't21_snapshots'][ind, :, 0 : params['crop'], 0 : params['crop']])
    else:
        #use everything!
        print('reading all data', len(ind))
        data = np.asarray(f['Data'][u't21_snapshots'][ind, :, :, :]) # (N_realizations, N_redshifts, N_pix, N_pix)
        #print('loaded data', len(ind))

    print('finished loading data.', data.shape)

    data  = data.transpose(0,2,3,1)  # (N_realizations, N_pix, N_pix, N_redshifts)

    return data, data[0].shape
    
def readImages(ind, **params):

    print('reading data from', inputFile)

    f = h5py.File(inputFile, 'r')

    #if params['debug'] == True:
    #    data = np.asarray(f['Data'][u't21_snapshots'][ind,:,0:16,0:16])
    if 'crop' in params:
        #print('cropping.')
        #use just the top corner of the images
        data = np.asarray(f['Data'][u't21_snapshots'][ind,:,0:params['crop'],0:params['crop']])
    else:
        #use everything!
        print('reading all data', len(ind))
        data = np.asarray(f['Data'][u't21_snapshots'][ind,:,:,:]) # (N_realizations, N_redshifts, N_pix, N_pix)
    #print('loaded data', len(ind))

    print('finished loading data.', data.shape)

    data  = data.transpose(0,2,3,1) #(N_realizations, N_pix, N_pix, N_redshifts)

    return data, data[0].shape


images1, shape = readImages(inputFile=inputFile1, ind=np.arange(1000))
images2, shape = readImages(inputFile=inputFile2, ind=np.arange(1000))

fig, axes = plt.subplots(figsize=(15,10),nrows=2, ncols=3, sharex=False, sharey=False)

for r, res in enumerate(results):
    #load reaults
    result=np.load(res)

    axes[0,0].scatter(result["truth"][:,n]/factor, result["prediction"][:,n]/factor,
    s=6, lw=0, alpha=0.9, label=fold[r])
    x = np.linspace(0.95*np.min(result["truth"][:,n]/factor),
                    1.05*np.max(result["truth"][:,n]/factor), 1000)
    axes[0,0].plot(x, x, 'k--',lw=5,alpha=0.2)


    axes[1,0].plot(x-x, 'k--',lw=5,alpha=0.2)
    relative_error = (result["prediction"][:,n]/factor-result["truth"][:,n]/factor)/result["truth"][:,n]
    axes[1,0].scatter(result["truth"][:,n]/factor, relative_error,s=6, lw=0, alpha=0.9, label=fold[r])

# plot box plot
axes[0,1].boxplot(df["Slope"], labels=" ")
axes[0,1].set_ylabel("")
axes[0,1].set_xlabel("Best Fit Model")

# Plot distance from mean
axes[1,1].plot(np.arange(1,11),coeff,'o')
axes[1,1].plot(np.arange(1,11),mean_,'k--',lw=1,markersize=4)
#axes[1,1].text(5.0, 0.2, r'$\mu$ vector [b_0 , b_1]: '+str(mean_),
#            {'color': 'blue', 'fontsize': 10, 'ha': 'center', 'va': 'center',
#            'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})
for indx, coef in enumerate(coeff[:,1]):
    diff = (coef-mean_[0,1])/2
    axes[1,1].vlines(indx+1,coef,mean_[0,1],colors='r',linestyles= 'dashed')
    axes[1,1].text(indx+1.25, coef-diff, r'%.2f' % diff,
    {'color': 'red', 'fontsize': 10, 'ha': 'center', 'va': 'center'})
axes[1,1].set_xlabel("Fold Number")
axes[1,1].set_ylabel("Distance from Mean")

axes[1,0].set_xlabel('True')
axes[0,0].set_xlim(0.028,0.1)
axes[1,0].set_xlim(0.028,0.1)
axes[0,0].set_xscale('linear')
axes[1,0].set_xscale('linear')

axes[0,0].set_ylabel('Predicted')
axes[0,0].set_ylim(0.028,0.1)
axes[1,0].set_ylabel('Residual Relative Difference')
axes[0,0].set_yscale('linear')
axes[1,0].set_yscale('linear')
 
axes[0,0].legend(markerscale=2.5)
plt.savefig("bestmodel_4residual.png")

