from __future__ import print_function, division, absolute_import

import numpy as np, matplotlib.pyplot as plt
from scipy import stats
import glob

n=0
factor=1000.

data_path = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/"
results = sorted(glob.glob(data_path+"results*.npy"))

def read_results(filename):

    #read in file
    result=np.load(filename)
    truth = (result["truth"][:,n]/factor)[:5*20]
    prediction = (result["prediction"][:,n]/factor)[:5*20]

    return [truth, prediction]


def plot_true_tau_stats(filename):
    x = read_results("/pylon5/as5phnp/tbilling/data/t21_snapshots_nowedge_v9.hdf5")[0]
    
    # Statistical Estimates
    nobs, minmax, mean, variance, skewness, kurtosis = stats.describe(x)
    stat_vlaues = np.array(minmax , mean, variance, skewness)
    
    plt.text(5.0, 0.2, r'$\mu$ vector [minmax , mean, variance, skewness]: '+str(stat_vlaues),
            {'color': 'blue', 'fontsize': 10, 'ha': 'center', 'va': 'center',
            'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})
    
    # putting labels
    plt.xlabel('True Optical Depth')
    plt.ylabel('Prediction')
    
    # Save plot
    plt.savefig(data_path+"true_tau_stats{}.png".format(str(fold)))
    plt.clf()

    return
    
def plot_stat_bins(x,y, fold):
    # x are the predicted values for a fold
    
    # Plotting the mean per bin
    bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=10)
    plt.plot(x,y, 'b.', label='Raw Data')
    plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='k', lw=5,label='Binned Statistic of Data')
    
    # Save plot
    plt.savefig(data_path+"mean_bin_stats_{}.png".format(str(fold)))
    plt.clf()
    
    # Plotting the standard deviation per bin
    bin_means, bin_edges, binnumber = stats.binned_statistic(x,y, statistic='std', bins=10)
    plt.plot(x,y, 'b.', label='Raw Data')
    plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='k', lw=5,label='Binned Statistic of Data')
    
    # Save plot
    plt.savefig(data_path+"std_bin_stats_{}.png".format(str(fold)))
    plt.clf()
    
    return
    

def main(filename,fold):
    # Read in observations
    x = read_results(filename)[0]
    y = read_results(filename)[1]

    # PLot bin statistics
    plot_stat_bins(x=x,y=y, fold=fold)
    return

if __name__ == "__main__":

    for r, res in enumerate(results):
        main(filename=res,fold=r)
    #plot_true_tau_stats()

