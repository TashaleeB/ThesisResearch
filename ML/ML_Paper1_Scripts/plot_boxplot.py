import os, h5py, io
import numpy as np, matplotlib.pyplot as plt, pandas as pd

data_path = '/pylon5/as5phnp/tbilling/data/'
v1_no_w = data_path+'t21_snapshots_nowedge_v9.hdf5'
v2_w = data_path+'t21_snapshots_wedge_v9.hdf5'


u = """Intercept,Slope,
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


data = io.StringIO(u)
# index_col = False to include both
# index_col = 1 to include intercept
df = pd.read_csv(data, sep=",", index_col=0)
plt.clf()
plt.figure(figsize=(10,10))
plt.boxplot(df["Slope"], labels=" ")
plt.ylabel("")
plt.xlabel("Best Fit Model")
#plt.boxplot(df["Intercept"])
plt.savefig("bestmodel_box_plot.png")
plt.show()

