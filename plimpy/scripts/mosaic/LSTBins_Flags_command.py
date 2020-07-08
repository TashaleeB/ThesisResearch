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


night1 = sorted(glob.glob("/lustre/aoc/projects/hera/H1C_IDR2/2458098/zen.2458098.4*.xx.HH.uvOR"))
night2 = sorted(glob.glob("/lustre/aoc/projects/hera/H1C_IDR2/2458099/zen.2458099.4*.xx.HH.uvOR"))
night3 = sorted(glob.glob("/lustre/aoc/projects/hera/H1C_IDR2/2458101/zen.2458101.4*.xx.HH.uvOR"))
night4 = sorted(glob.glob("/lustre/aoc/projects/hera/H1C_IDR2/2458102/zen.2458102.4*.xx.HH.uvOR"))

uvd1 = UVData()
uvd1.read_miriad(night1[0:3])

uvd2 = UVData()
uvd2.read_miriad(night2[0:3])

uvd3 = UVData()
uvd3.read_miriad(night3[0:3])

uvd4 = UVData()
uvd4.read_miriad(night4[0:3])



# load data and meta data. LST arrays are the lst1, lst2, lst3 variables
data1, flgs1, ap1, a1, f1, t1, lst1, p1 = hc.io.load_vis(uvd1, return_meta=True)
data2, flgs2, ap2, a2, f2, t2, lst2, p2 = hc.io.load_vis(uvd2, return_meta=True)
data3, flgs3, ap3, a3, f3, t3, lst3, p3 = hc.io.load_vis(uvd3, return_meta=True)
data4, flgs4, ap4, a4, f4, t4, lst4, p4 = hc.io.load_vis(uvd4, return_meta=True)

data_list = [data1, data2, data3,data4]
lst_list = [lst1, lst2, lst3,lst4]
flgs_list = [flgs1, flgs2, flgs3,flgs4]

(lst_bins, data_avg, lst_flags, data_std,data_num)=hc.lstbin.lst_bin(data_list=data_list, lst_list=lst_list, dlst=np.median(np.diff(lst1)), flags_list=
                                                                          flgs_list, flag_thresh=0.7)
# plot the data's native LST integrations
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

ax.yaxis.set_ticklabels(['','2458098', '2458099', '2458101','2458102'])

ax.set_xlabel("LST [radians]")
plt.show()
keys = lst_flags.keys()
for k, (bl1,bl2,p) in enumerate(keys[1:2]):

    plt.imshow((lst_flags[keys[k]]*1),aspect="auto", cmap='gray')
    plt.xlabel("Chan [Arb.]")
    plt.title("{}{}{}".format(bl1,bl2,p))
    plt.colorbar()
    #plt.savefig("./waterfall_flagvsLSTbin.{}_{}.{}.pdf".(bl1,bl2,p))
    plt.show()
float(np.sum(lst_flags[keys[0]]))/(lst_flags[keys[0]].flatten().shape[0])
print(time.asctime( time.localtime(time.time())))
night1 = sorted(glob.glob("/lustre/aoc/projects/hera/H1C_IDR2/2458098/zen.2458098.4*.xx.HH.uvOR"))
uvd1 = UVData()
uvd1.read_miriad(night1[0:3])
print(time.asctime( time.localtime(time.time())))

print(time.asctime( time.localtime(time.time())))
night1 = sorted(glob.glob("/lustre/aoc/projects/hera/H1C_IDR2/2458098/zen.2458098.4*.xx.HH.uvOR"))
uvd1 = UVData()
uvd1.read_miriad(night1[0:3])
print(time.asctime( time.localtime(time.time())))
data1, flgs1, ap1, a1, f1, t1, lst1, p1 = hc.io.load_vis(uvd1, return_meta=True)
print(time.asctime( time.localtime(time.time())))
(lst_bins, data_avg, lst_flags, data_std,data_num)=hc.lstbin.lst_bin(data_list=data_list, lst_list=lst_list, dlst=np.median(np.diff(lst1)), flags_list=flgs_list, flag_thresh=0.7)
print(time.asctime( time.localtime(time.time())))

print(time.asctime( time.localtime(time.time())))
(lst_bins_, data_avg_, lst_flags_, data_std_,data_num_)=hc.lstbin.lst_bin(data_list=[data1],
                                                                          lst_list=[lst1], dlst=np.median(np.diff(lst1)), flags_list=[flgs1], flag_thresh=0.7)
print(time.asctime( time.localtime(time.time())))



