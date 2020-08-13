from __future__ import print_function, division, absolute_import

import numpy as np, pandas as pd, io

n=0
factor=1000.

data_path = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/opt_versions/200steps_v9/" # No wedge filtering
#data_path = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/opt_versions/wedgefilter_v9/"# Wedge Filterd

results = sorted(glob.glob(data_path+"bestmodel_pred_results*.npy"))
fold = np.arange(0,len(results))
nbins = 10

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

# -----------------------------------------------

data = io.StringIO(u)

df = pd.read_csv(data, sep=",", index_col=0)


def std_(csv_data):
    # Read in observations
    data = csv_data
    
    residual = np.sum(np.array([(x_i-1.0)**2 for x_i in data]))/len(data)
    sigma = np.sqrt(residual)
    
    mean_val = np.mean(data)
    
    return sigma, mean_val

if __name__ == "__main__":

    std, mean =std_(csv_data=df["Slope"])
    print("Mean ", mean)
    print("STD ", std)
    


