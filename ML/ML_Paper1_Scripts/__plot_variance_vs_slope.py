from __future__ import print_function, division, absolute_import

import numpy as np, matplotlib.pyplot as plt
import glob

n=0
factor=1000.

scores = []
param_count = []
coefficients = []
eval_scores_names = []

data_path = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/opt_versions/wedgefilter_v9/"
results = sorted(glob.glob(data_path+"bestmodel_pred_results*.npy"))
eval_score_files = data_path+"eval_score_*data.npz"

#Force the file names to be in ascending order.
file_names = sorted(glob.glob(eval_score_files))
for s in range(len(file_names)):
    eval_scores_names.append(data_path+'eval_score_{}data.npz'.format(s+1))

eval_scores= np.array(eval_scores_names)
del(file_names)

def read_results(filename):

    #read in file
    result=np.load(filename)
    truth = (result["truth"][:,n]/factor)[:5*20]
    prediction = (result["prediction"][:,n]/factor)[:5*20]

    return [truth, prediction]


def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x

    return [b_0, b_1]

# Plot variance vs slope
def var_vs_slope_plot(coefficient):
    coefficients = np.array(coefficient,dtype=np.float64)
    plt.plot(np.arange(1,11),coefficients,'o')
    variance = np.var()
    plt.plot(np.arange(1,11),mean,'k--',lw=1,markersize=4)
    plt.text(5.0, 0.2, r'$\mu$ vector [b_0 , b_1]: '+str(mean_value),
            {'color': 'blue', 'fontsize': 10, 'ha': 'center', 'va': 'center',
            'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})
            
    plt.xlabel("Fold Number")
    plt.ylabel("Distance from Mean")
    plt.savefig(data_path+"bestmodel_residual_of_mean.png")
    plt.clf()

def main(filename, eval_score_files, fold):
    # Read in observations
    x = read_results(filename)[0]
    y = read_results(filename)[1]

    # estimating coefficients
    b = estimate_coef(x, y)
    coefficients.append(b)
    print("Estimated coefficients:\nb_0 = {} \
        \nb_1 = {}".format(b[0], b[1]))
    
    # Load the evaluation scores
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
    var_sorted_scores = np.var(sorted_scores, axis=0)
    return var_sorted_scores, coefficients

if __name__ == "__main__":

for r, res in enumerate(results):
    variance_, coefficient_ = main(filename=results[i], eval_score_files=eval_scores_names[i], fold=i)
var_vs_slope_plot(coefficient=coefficients)
