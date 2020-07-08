#!/bin/sh

#  plot_JDvsLSTBin.py
#  
#
#  Created by Tashalee Billings on 08/29/19.
#

#printf "\e[?2004l"
import numpy as np
import hera_cal as hc
import argparse,os, glob, sys, operator
import matplotlib.pyplot as plt

import progress_bar # Cool thing James wrote.

from pyuvdata import UVData, UVCal, utils
from hera_cal.io import load_cal

from IPython.display import clear_output
from copy import deepcopy

# Define Arguments
a = argparse.ArgumentParser(description="Run using python as: python plot_JDvsLSTBin.py <args>")
a.add_argument('--data_path', '-dp', type=str, help='Path to data. eg. "/lustre/aoc/projects/hera/H1C_IDR2/"',required=True)
a.add_argument('--juliandate', '-jd', type=str, help='The JD you are interested in. eg ["2458098.4","2458099.4","2458101.4","2458102.4"]',required=True)
a.add_argument('--pol', '-pol', type=str, default='xx', help='The polarization you are interested in. eg "xx" or "yy"',required=True)
a.add_argument('--extention', '-ext', type=str, help='Extention of the miriad data you are working with. eg "uvOR"',required=True)

# Load and Configure Data
if __name__ == '__main__':

args = a.parse_args()

# Parse Arguments
data_path = args.data_path # "/lustre/aoc/projects/hera/H1C_IDR2/"
juliandate_list = args.juliandate # "2458099.4"
pol = args.pol # "xx"
extention = args.extention # "uvOR"

# Check to make sure that you provided a list of Julian Dates
if type(juliandate_list)= list:
raise TypeError("Your Julian Data has a type {} but it should be a list.".format(type(juliandate_list)))

# Number of nights
num_nights = len(juliandate_list)
uvds = []

_uvds = []
data_list = []
lst_list = []
flgs_list = []

for juliandate in juliandate_list:

night = sorted(glob.glob(os.path.join(data_path, "zen."+juliandate+"*."+pol+".HH."+extention)))
print("Number of files: ", len(night))

# Read data
_uvd = UVData()
_uvd.read(night)

# load data and meta data. LST arrays are the lst1, lst2, lst3 variables
data, flgs, ap, a, f, t, lst, p = hc.io.load_vis(_uvd, return_meta=True)

# Append to List
_uvds.append(_uvd)

# concatenate source files
uvd = reduce(operator.add, _uvds)

# Add data to list
data_list.append(data)
lst_list.append(lst)
flgs_list.append(flgs)

# get integration duration in radians
# 0.0007829849986082937 -> 0.0007829849986082937*60*60=2.8187459949898575 sec
delta_lst = np.median(np.diff(lst)) # [rad]

(lst_bins, data_avg,
lst_flags, data_std,
data_num) = hc.lstbin.lst_bin(data_list=data_list,lst_list=lst_list,
dlst=delta_lst, flag_thresh=0.7,
flags_list=flgs_list)





# Percentage of data flagged for each LST Bin over several days.
mask_file_name = outpath+'zen.'+lststr(ra0*u.rad)+'_'+lststr((ra0+interval)*u.rad)
mask_file = open("percentage_flag_data.txt", "w")
mask_file.write("#CRTFv0\n")
mask_file.writelines(["circle[[" + str(x[0]) + "deg, " + str(x[1]) + "deg], 0.5deg]\n" for x in in_bounds])
mask_file.writelines(["circle[[" + str(x[0]) + "deg, " + str(x[1]) + "deg], 0.5deg]\n" for x in fornax_inb])
mask_file.close()




In [14]:
...: night1 = sorted(glob.glob("/lustre/aoc/projects/hera/H1C_IDR2/2458098/zen.2458098.4*.xx.HH.uvOR"))
...: night2 = sorted(glob.glob("/lustre/aoc/projects/hera/H1C_IDR2/2458099/zen.2458099.4*.xx.HH.uvOR"))
...: night3 = sorted(glob.glob("/lustre/aoc/projects/hera/H1C_IDR2/2458101/zen.2458101.4*.xx.HH.uvOR"))
...: night4 = sorted(glob.glob("/lustre/aoc/projects/hera/H1C_IDR2/2458102/zen.2458102.4*.xx.HH.uvOR"))
...:
...: uvd1 = UVData()
...: uvd1.read(night1[0:3])
...:
...: uvd2 = UVData()
...: uvd2.read(night2[0:3])
...:
...: uvd3 = UVData()
...: uvd3.read(night3[0:3])
...:
...: uvd4 = UVData()
...: uvd4.read(night4[0:3])
...:
...:
...:
...: # load data and meta data. LST arrays are the lst1, lst2, lst3 variables
...: data1, flgs1, ap1, a1, f1, t1, lst1, p1 = hc.io.load_vis(uvd1, return_meta=True)
...: data2, flgs2, ap2, a2, f2, t2, lst2, p2 = hc.io.load_vis(uvd2, return_meta=True)
...: data3, flgs3, ap3, a3, f3, t3, lst3, p3 = hc.io.load_vis(uvd3, return_meta=True)
...: data4, flgs4, ap4, a4, f4, t4, lst4, p4 = hc.io.load_vis(uvd4, return_meta=True)
...:
...: data_list = [data1, data2, data3,data4]
...: lst_list = [lst1, lst2, lst3,lst4]
...: flgs_list = [flgs1, flgs2, flgs3,flgs4]
...:
...: (lst_bins, data_avg, lst_flags, data_std,data_num)=hc.lstbin.lst_bin(data_list=data_list, lst_list=lst_list, dlst=np.median(np.diff(lst1)), flags_list =flgs_list, flag_thresh=0.7)
...:

# Make a Waterfall Plot of the Flagged data for a given LST Bin
In [42]: for k, (bl1,bl2,p) in enumerate(keys[0:1]):
...:
...:     plt.imshow((lst_flags[keys[k]]*1),aspect="auto", cmap='gray')
...:     plt.xlabel("Chan [Arb.]")
...:     plt.title("({},{},{})".format(bl1,bl2,p))
...:     plt.colorbar()
...:     #plt.savefig("./waterfall_flagvsLSTbin.{}_{}.{}.pdf".(bl1,bl2,p))
...:     plt.show()
...:



# Make a Plot of JD vs LST
...: fig, ax = plt.subplots(1, 1, figsize=(15, 8))
...: ax.grid()
...: p1, = ax.plot(lst1, np.ones_like(lst1)*0, color='darkred', ms=15, marker='|', ls='')
...: p2, = ax.plot(lst2, np.ones_like(lst2)*1, color='darkorange', ms=15, marker='|', ls='')
...: p3, = ax.plot(lst3, np.ones_like(lst3)*2, color='steelblue', ms=15, marker='|', ls='')
...: p4, = ax.plot(lst4, np.ones_like(lst4)*3, color='blue', ms=15, marker='|', ls='')

...: ax.set_ylim(-1, len(data_list))

...: ax.xaxis.set_minor_locator(AutoMinorLocator())
...: ax.tick_params(which='both', width=2)
...: ax.tick_params(which='major', length=7)
...: ax.tick_params(which='minor', length=4, color='r')

...: ax.xaxis.grid(True, which='minor')
...: ax.yaxis.grid(True, which='minor')
...:
...: ax.yaxis.set_ticklabels(['','2458098', '2458099', '2458101','2458102'])
...:
...: ax.set_xlabel("LST [radians]")
...: plt.show()
...:
