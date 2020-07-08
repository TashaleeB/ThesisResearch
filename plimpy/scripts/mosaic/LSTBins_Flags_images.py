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

# List of JD nights
jd1 = '2458098'
jd2 = '2458099'
jd3 = '2458101'
jd4 = '2458102'

night1 = sorted(glob.glob("/lustre/aoc/projects/hera/H1C_IDR2/{}/zen.{}.4*.xx.HH.uvOR".format(jd1,jd1)))
night2 = sorted(glob.glob("/lustre/aoc/projects/hera/H1C_IDR2/{}/zen.{}.4*.xx.HH.uvOR".format(jd2,jd2)))
night3 = sorted(glob.glob("/lustre/aoc/projects/hera/H1C_IDR2/{}/zen.{}.4*.xx.HH.uvOR".format(jd3,jd3)))
night4 = sorted(glob.glob("/lustre/aoc/projects/hera/H1C_IDR2/{}/zen.{}.4*.xx.HH.uvOR".format(jd4,jd4)))

# Read in data from those nights
# Load data and meta data. LST arrays are the lst1, lst2, lst3 variables
print("Reading in night1",time.asctime(time.localtime(time.time())))
uvd1 = UVData()
uvd1.read(night1)
print("Reading in meta data for night1",time.asctime(time.localtime(time.time())))
data1, flgs1, ap1, a1, f1, t1, lst1, p1 = hc.io.load_vis(uvd1, return_meta=True)
del(uvd1)

print("Reading in night2",time.asctime(time.localtime(time.time())))
uvd2 = UVData()
uvd2.read(night2)
print("Reading in meta data for night2",time.asctime(time.localtime(time.time())))
data2, flgs2, ap2, a2, f2, t2, lst2, p2 = hc.io.load_vis(uvd2, return_meta=True)
del(uvd2)

print("Reading in night3",time.asctime(time.localtime(time.time())))
uvd3 = UVData()
uvd3.read(night3)
print("Reading in meta data for night3",time.asctime(time.localtime(time.time())))
data3, flgs3, ap3, a3, f3, t3, lst3, p3 = hc.io.load_vis(uvd3, return_meta=True)
del(uvd3)

print("Reading in night4",time.asctime(time.localtime(time.time())))
uvd4 = UVData()
uvd4.read(night4)
print("Reading in meta data for night4",time.asctime(time.localtime(time.time())))
data4, flgs4, ap4, a4, f4, t4, lst4, p4 = hc.io.load_vis(uvd4, return_meta=True)
print(time.asctime(time.localtime(time.time())))
del(uvd4)

# Make list of the data & meta data to
data_list = [data1, data2, data3,data4]
lst_list = [lst1, lst2, lst3,lst4]
flgs_list = [flgs1, flgs2, flgs3,flgs4]

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

for k, (bl1,bl2,p) in enumerate(keys[0:3]):
    # Build grid
    numf = data_avg[keys[k]].shape[1]
    X, Y = np.meshgrid(np.linspace(100, 200, len(np.arange(0,numf)[0::8]), endpoint=False), lst_bins[::4])
    X = X.ravel()
    Y = Y.ravel()

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.grid()

    cax = ax.scatter(X, Y, c=data_num[keys[k]][::4, ::8].ravel(), s=30, cmap='viridis')
    cbar = fig.colorbar(cax)
    cbar.set_ticks([0,1,2,3])
    cbar.set_label('number of points in each LST bin', fontsize=15)
    ax.set_xlabel('Frequency [MHz]', fontsize=15)
    ax.set_ylabel('LST [radians]', fontsize=15)
    plt.savefig("/lustre/aoc/projects/hera/tbilling/images/{}_{}_{}_DataPointPerLSTFreqBin.pdf".format(bl1,bl2,p))
    plt.close()
    #plt.show()

for k, key in enumerate(keys[0:3]):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    bl1,bl2,p = key
    ax = axes[0,0]
    plt.sca(ax)
    d = data1[key].copy()
    d[flgs1[key]] *= np.nan
    plt.imshow(np.abs(d), aspect="auto")
    cbar = plt.colorbar()
    cbar.set_label('Night 1 Amplitude', fontsize=12)
    ax.set_title('night1 {} AMP'.format(key), fontsize=12)
    ax.set_ylabel('LST Integrations', fontsize=12)
    ax.set_ylim(224, 0)

    ax = axes[0,1]
    plt.sca(ax)
    d = data2[key].copy()
    d[flgs2[key]] *= np.nan
    plt.imshow(np.abs(d), aspect="auto")
    cbar = plt.colorbar()
    cbar.set_label('Night 2 Amplitude', fontsize=12)
    ax.set_title('night2 {} AMP'.format(key), fontsize=12)
    ax.set_ylabel('LST Integrations', fontsize=12)
    ax.set_ylim(224, 0)

    ax = axes[1,0]
    plt.sca(ax)
    d = data3[key].copy()
    d[flgs3[key]] *= np.nan
    plt.imshow(np.abs(d), aspect="auto")
    cbar = plt.colorbar()
    cbar.set_label('Night 3 Amplitude', fontsize=12)
    ax.set_title('night3 {} AMP'.format(key), fontsize=12)
    ax.set_xlabel('Chan [Arb.]', fontsize=12)
    ax.set_ylabel('LST Integrations', fontsize=12)
    ax.set_ylim(224, 0)

    ax = axes[1,1]
    plt.sca(ax)
    d = data4[key].copy()
    d[flgs4[key]] *= np.nan
    plt.imshow(np.abs(d), aspect="auto")
    cbar = plt.colorbar()
    cbar.set_label('Night 4 Amplitude', fontsize=12)
    ax.set_title('night4 {} AMP'.format(key), fontsize=12)
    ax.set_xlabel('Chan [Arb.]', fontsize=12)
    ax.set_ylabel('LST Integrations', fontsize=12)
    ax.set_ylim(224, 0)

    ax = axes[0,2]
    plt.sca(ax)
    d = data_avg[key].copy()
    d[lst_flags[key]] *= np.nan
    plt.imshow(np.abs(d), aspect="auto")
    cbar = plt.colorbar()
    cbar.set_label('Amplitude', fontsize=14)
    ax.set_title('Avg Data Per LST Bin for {}'.format(key), fontsize=14)
    ax.set_xlabel('Chan [Arb.]', fontsize=14)
    ax.set_ylabel('LST Integrations', fontsize=14)
    
    plt.tight_layout()

    plt.savefig("/lustre/aoc/projects/hera/tbilling/images/waterfall.{}_{}.{}.pdf".format(bl1,bl2,p))
    plt.close()
