from __future__ import print_function, division, absolute_import

import numpy as np, matplotlib.pyplot as plt
from scipy import stats
import glob

n=0
factor=1000.

data_path = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/opt_versions/200steps_v9/"
#data_path = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/opt_versions/wedgefilter_v9/"
results = sorted(glob.glob(data_path+"bestmodel_pred_results*.npy"))
fold = np.arange(0,len(results))
nbins = 10

def read_results(filename):

    #read in file
    result=np.load(filename)
    truth = (result["truth"][:,n]/factor)[:5*20]
    prediction = (result["prediction"][:,n]/factor)[:5*20]

    return [truth, prediction]

def plot_stat_bins(x, y, indx):
    # x are the predicted values
    
    # Plotting the std per bin
    bin_std = x[:,indx:indx+nbins].std()
    bin_edges = y[:,indx:indx+nbins]
    
    for idx in range(0,nbins):
        #plt.plot(x[idx, indx:indx+nbins],y[idx, indx:indx+nbins], 'b.')#, label='Raw Data')
        plt.scatter(y[idx, indx:indx+nbins], x[idx, indx:indx+nbins], s=6, lw=0, alpha=0.9, c='blue')
    plt.hlines(bin_std, bin_edges.min(), bin_edges.max(), colors='k', lw=5)#,label='Binned Statistic of Data')
    
    # Save plot
    plt.ylabel("Binned Standard Deviation ")
    plt.xlabel("Optical Depth")
    
    plt.ylim(-0.0005, 0.0005)
    plt.savefig(data_path+"std_bin_stats.png")
    plt.clf()
    
    return
    

def main(filename):
    # Read in observations
    x = []
    y = []
    
    for f in filename:
        data_x = np.array(sorted(read_results(f)[0]))
        data_y = np.array(sorted(read_results(f)[1]))
        relative_error = -(data_x - data_y)/data_y
        x.append(relative_error/factor)
        y.append(data_y)
    x = np.array(x)
    y = np.array(y)

    # Plot bin statistics
    plt.figure(figsize=(15,10))
    for i in range(0,y.shape[1], nbins):
        plot_stat_bins(x=x, y=y, indx=i)
    
    return

if __name__ == "__main__":

    main(filename=results)


