from __future__ import print_function, division, absolute_import

import numpy as np, matplotlib.pyplot as plt
import hera_cal as hc
import argparse,os, glob, sys, operator, time, subprocess
import matplotlib.pyplot as plt

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
from pyuvdata import UVData, UVCal, utils
from hera_cal.io import load_cal
from IPython.display import clear_output
from copy import deepcopy

path = "/lustre/aoc/projects/hera/H1C_IDR2/"

#list_of_jds = sorted(glob.glob(path+"2458*/"))
jds = [jdfile.split("/")[-2] for jdfile in sorted(glob.glob(path+"2458*/"))]

all_nights = []
for jd in jds:
    night = sorted(glob.glob(path+"{}/zen.{}.4*.xx.HH.uvOR".format(jd,jd)))
    all_nights+=night

length = len(all_nights)

print("Reading All Nights",time.asctime(time.localtime(time.time())))
uvd_all_nights = UVData()
uvd_all_nights.read(all_nights[0:round(length/2)])
print("Reading in meta data for all nights",time.asctime(time.localtime(time.time())))
data, flgs, ap, a, f, t, lst, p = hc.io.load_vis(uvd_all_nights, return_meta=True)
del(uvd_all_nights)
del(ap,a,f,t,p)

# each file is data[60,1024] lst[60,]

# Make list of the data & meta data to
data_list = [data]
lst_list = [lst]
flgs_list = [flgs]

# LST bin with some flag threshold
bin_width = np.median(np.diff(lst1)) # radians
bin_width_min = 0.0007830490163484*(180/np.pi)*60
bin_width_sec = 0.0007830490163484*(180/np.pi)*60*60
print("Starting with bin width {0:.4f} min".format(bin_width_min))
print("Starting with bin width {0:.4f} sec".format(bin_width_sec))

print("Starting LST Bin",time.asctime(time.localtime(time.time())))
(lst_bins, data_avg, lst_flags,data_std,data_num)=hc.lstbin.lst_bin(data_list=data_list, lst_list=lst_list, dlst=bin_width, flags_list=flgs_list, flag_thresh=0.7)
print("Ending LAST Bin",time.asctime(time.localtime(time.time())))

# Plot the data's native LST integrations
fig, ax = plt.subplots(1, 1, figsize=(15, 8))
ax.grid()
p1, = ax.plot(lst1, np.ones_like(lst1)*0, color='darkred', ms=15, marker='|', ls='')
p2, = ax.plot(lst2, np.ones_like(lst2)*1, color='darkorange', ms=15, marker='|', ls='')
p3, = ax.plot(lst3, np.ones_like(lst3)*2, color='steelblue', ms=15, marker='|', ls='')
p4, = ax.plot(lst4, np.ones_like(lst4)*3, color='blue', ms=15, marker='|', ls='')

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='both', width=2)
ax.tick_params(which='major', length=7)
ax.tick_params(which='minor', length=4, color='r')
ax.set_ylim(-1, len(data_list))
ax.xaxis.grid(True, which='minor')
ax.yaxis.grid(True, which='minor')

ax.yaxis.set_ticklabels(['',jd1, jd2, jd3, jd4])
ax.set_xlabel("LST [radians]")
ax.set_ylabel("Julian Date")

plt.savefig("/lustre/aoc/projects/hera/tbilling/images/lst_integrattions.pdf")
plt.close()
#plt.show()

# PLot of the number of data that fall into each LST Bin
keys = list(lst_flags.keys())


subtitle_string = "Percentage of Data Flagged {}"
for k, key in enumerate(keys[153:153+15]):
    fig, axes = plt.subplots(1,1, figsize=(18, 12))
    ax = axes[0,0]
    plt.sca(ax)
    
    bl1,bl2,p = key
    
    perc_of_data_flagged = float(sum(flgs[key].flatten()))/((flgs[key].flatten()).shape[0])*100.
    d = data[key].copy()
    d[flgs[key]] *= np.nan
    plt.imshow(np.abs(d), aspect="auto")
    cbar = plt.colorbar()
    cbar.set_label('Night Amplitude', fontsize=12)
    ax.set_title('Ant {} AMP: Percentage Flagged {}'.format(key,perc_of_data_flagged), fontsize=12)
    ax.set_ylabel('LST Integrations', fontsize=12)
    #ax.set_ylim(224, 0)
    
    plt.tight_layout()

    plt.savefig("/lustre/aoc/projects/hera/tbilling/images/waterfall_PercDataFlagged.{}_{}.{}.pdf".format(bl1,bl2,p))
    plt.close()
