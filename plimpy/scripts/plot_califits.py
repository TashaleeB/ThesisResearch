#!/bin/sh

#  plot_califits.py
#  
#
#  Created by Tashalee Billings on 4/13/18.
#

#printf "\e[?2004l"
import numpy as np
import argparse,os, glob, sys, copy,operator
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pyuvdata import UVCal, UVData
from hera_cal.io import load_cal


# If for some reason you get a module error when you try to import io.py then use the function load_cal directly

import numpy as np
from pyuvdata import UVCal, UVData
from pyuvdata import utils as uvutils
from collections import OrderedDict as odict
from hera_cal.datacontainer import DataContainer
import hera_cal as hc
import operator
import os
import copy


polnum2str = {-5: "xx", -6: "yy", -7: "xy", -8: "yx"}
polstr2num = {"xx": -5, "yy": -6 ,"xy": -7, "yx": -8}

jonesnum2str = {-5: 'x', -6: 'y'}
jonesstr2num = {'x': -5, 'y': -6}

def load_cal(input_cal, return_meta=False):
    '''Load calfits files or UVCal objects into dictionaries, optionally returning
        the most useful metadata. More than one spectral window is not supported.
        Arguments:
        input_cal: path to calfits file, UVCal object, or a list of either
        return_meta: if True, returns additional information (see below)
        Returns:
        if return_meta is True:
        (gains, flags, quals, total_qual, ants, freqs, times, pols)
        else:
        (gains, flags)
        gains: Dictionary of complex calibration gains as a function of time
        and frequency with keys in the (1,'x') format
        flags: Dictionary of flags in the same format as the gains
        quals: Dictionary of of qualities of calibration solutions in the same
        format as the gains (e.g. omnical chi^2 per antenna)
        total_qual: ndarray of toal calibration quality for the whole array
        (e.g. omnical overall chi^2)
        ants: ndarray containing unique antenna indices
        freqs: ndarray containing frequency channels (Hz)
        times: ndarray containing julian date bins of data
        pols: list of antenna polarization strings
        '''
    #load UVCal object
    cal = UVCal()
    if isinstance(input_cal, (tuple, list, np.ndarray)): #List loading
        if np.all([isinstance(ic, str) for ic in input_cal]): #List of calfits paths
            cal.read_calfits(list(input_cal))
        elif np.all([isinstance(ic, UVCal) for ic in input_cal]): #List of UVCal objects
            cal = reduce(operator.add, input_cal)
        else:
            raise TypeError('If input is a list, it must be only strings or only UVCal objects.')
    elif isinstance(input_cal, str): #single calfits path
        cal.read_calfits(input_cal)
    elif isinstance(input_cal, UVCal): #single UVCal object
        cal = input_cal
    else:
        raise TypeError('Input must be a UVCal object, a string, or a list of either.')
    
    #load gains, flags, and quals into dictionaries
    gains, quals, flags, total_qual = odict(), odict(), odict(), odict()
    for ip, pol in enumerate(cal.jones_array):
        if cal.total_quality_array is not None:
            total_qual[jonesnum2str[pol]] = cal.total_quality_array[0, :, :, ip].T
        else:
            total_qual = None
        for i, ant in enumerate(cal.ant_array):
            gains[(ant, jonesnum2str[pol])] = cal.gain_array[i, 0, :, :, ip].T
            flags[(ant, jonesnum2str[pol])] = cal.flag_array[i, 0, :, :, ip].T
            quals[(ant, jonesnum2str[pol])] = cal.quality_array[i, 0, :, :, ip].T

#return quantities
    if return_meta:
        ants = cal.ant_array
        freqs = np.unique(cal.freq_array)
        times = np.unique(cal.time_array)
        pols = [jonesnum2str[j] for j in cal.jones_array]
        return gains, flags, quals, total_qual, ants, freqs, times, pols
    else:
        return gains, flags
#%%
#----------------------------------------------------------
# Load and orgaize data
#----------------------------------------------------------
path="/lustre/aoc/projects/hera/H1C_IDR2/IDR2_1/2458115/zen.2458115."

xsmooth = glob.glob(path+'*.xx.HH.uv.smooth_abs.calfits')
ysmooth = glob.glob(path+'*.yy.HH.uv.smooth_abs.calfits')
xabs = glob.glob(path+'*.xx.HH.uv.abs.calfits')
yabs = glob.glob(path+'*.yy.HH.uv.abs.calfits')

xomni = glob.glob(path+'*.xx.HH.uv.omni.calfits')
yomni = glob.glob(path+'*.yy.HH.uv.omni.calfits')
xfirst = glob.glob(path+'*.xx.HH.uv.first.calfits')
yfirst = glob.glob(path+'*.yy.HH.uv.first.calfits')

print("Now Beginning to load data ...")

xgain_smooth, xflag_smooth = load_cal(xsmooth,return_meta=False)
print("Finsished Smoothcal")
xgain_abs, xflag_abs = load_cal(xabs,return_meta=False)
print("Finsished abscal")
xgain_omni, xflag_omni = load_cal(xomni,return_meta=False)
print("Finsished finsished omnical")
xgain_first, xflag_first = load_cal(xfirst,return_meta=False)
print("Finsished finished firstcal")

ygain_smooth, yflag_smooth = load_cal(ysmooth,return_meta=False)
print("Finsished Smoothcal")
ygain_abs, yflag_abs = load_cal(yabs,return_meta=False)
print("Finsished abscal")
ygain_omni, yflag_omni = load_cal(yomni,return_meta=False)
print("Finsished omnical")
ygain_first, yflag_first = load_cal(yfirst,return_meta=False)
print("Finsished firstcal")

xkey = xgain_smooth.keys()
ykey = ygain_smooth.keys()
#--------------------------------------------
cc=xgain_smooth[(1, 'x')]
cc[xflag_smooth[(1, 'x')]] *= np.nan #removing the bad data

plt.imshow(np.abs(cc), aspect='auto', interpolation='nearest') #for Gain Amplitude
#plt.imshow(np.angle(cc), aspect='auto', interpolation='nearest', vmin=-np.pi, vmax=np.pi) #for Phase
plt.title("zen.2458098.*.xx.HH.uv.smooth_abs.calfits: (1, 'X')")
plt.xlabel("Freq Channel")
plt.ylabel("Time Integrations")
plt.colorbar()
plt.show()

#OR To make 4 types of plots for one type of calfits file
k,i=0,0
xfilename = "zen.2458115.*.xx.HH.uv.smooth_abs.calfits"
xfucngain = xgain_smooth
fig, axes = plt.subplots(2, 2, figsize=(14, 15))
gs = gridspec.GridSpec(2,2)
fig.suptitle("Calfits Amplitude Waterfalls for East Pol & file {}".format(xfilename), fontsize=14)

iarr = [0,0,1,1]
jarr = [0,1,0,1]

for i,k in enumerate(xkey[0:len(iarr)]):
    ax=plt.subplot(gs[iarr[i],jarr[i]])
    ax.set_title(k, fontsize=12, y=1.01)
    rfi_flagdata = xfucngain[k].copy()
    rfi_flagdata[xflag_smooth[k]] *= np.nan
    ax.imshow(np.abs(rfi_flagdata), aspect='auto', interpolation='nearest')

    if jarr[i] == 0:
        [t.set_fontsize(10) for t in ax.get_yticklabels()]
        ax.set_ylabel('time integrations', fontsize=10)
    if iarr[i] == 1:
        [t.set_fontsize(10) for t in ax.get_xticklabels()]
        ax.set_xlabel('freq channel', fontsize=10)

fig.savefig('/lustre/aoc/projects/hera/tbilling/polim/calibration/plots/4xGAIN_{}.png'.format(xfilename))
#plt.show()

k,i=0,0
yfilename = "zen.2458115.*.yy.HH.uv.smooth_abs.calfits"
yfucngain = ygain_smooth
fig, axes = plt.subplots(2, 2, figsize=(14, 15))
gs = gridspec.GridSpec(2,2)
fig.suptitle("Calfits Amplitude Waterfalls for North Pol & file {}".format(yfilename), fontsize=14)

iarr = [0,0,1,1]
jarr = [0,1,0,1]

for i,k in enumerate(ykey[0:len(iarr)]):
    ax=plt.subplot(gs[iarr[i],jarr[i]])
    ax.set_title(k, fontsize=12, y=1.01)
    rfi_flagdata = yfucngain[k].copy()
    rfi_flagdata[yflag_smooth[k]] *= np.nan
    ax.imshow(np.abs(rfi_flagdata), aspect='auto', interpolation='nearest')
    
    if jarr[i] == 0:
        [t.set_fontsize(10) for t in ax.get_yticklabels()]
        ax.set_ylabel('time integrations', fontsize=10)
    if iarr[i] == 1:
        [t.set_fontsize(10) for t in ax.get_xticklabels()]
        ax.set_xlabel('freq channel', fontsize=10)

fig.savefig('/lustre/aoc/projects/hera/tbilling/polim/calibration/plots/4yGAIN_{}.png'.format(yfilename))
#plt.show()

k,i=0,0

fig, axes = plt.subplots(2, 2, figsize=(14, 15))
gs = gridspec.GridSpec(2,2)
fig.suptitle("Calfits Phase Waterfalls for East Pol & file {}".format(xfilename), fontsize=14)

iarr = [0,0,1,1]
jarr = [0,1,0,1]

for i,k in enumerate(xkey[0:len(iarr)]):
    ax=plt.subplot(gs[iarr[i],jarr[i]])
    ax.set_title(k, fontsize=12, y=1.01)
    rfi_flagdata = xfucngain[k].copy()
    rfi_flagdata[xflag_smooth[k]] *= np.nan
    ax.imshow(np.angle(rfi_flagdata), aspect='auto', interpolation='nearest', vmin=-np.pi, vmax=np.pi)
    
    if jarr[i] == 0:
        [t.set_fontsize(10) for t in ax.get_yticklabels()]
        ax.set_ylabel('time integrations', fontsize=10)
    if iarr[i] == 1:
        [t.set_fontsize(10) for t in ax.get_xticklabels()]
        ax.set_xlabel('freq channel', fontsize=10)

fig.savefig('/lustre/aoc/projects/hera/tbilling/polim/calibration/plots/4xPHASE_{}.png'.format(xfilename))
#plt.show()

k,i=0,0

fig, axes = plt.subplots(2, 2, figsize=(14, 15))
gs = gridspec.GridSpec(2,2)
fig.suptitle("Calfits Phase Waterfalls for North Pol & file {}".format(yfilename), fontsize=14)

iarr = [0,0,1,1]
jarr = [0,1,0,1]

for i,k in enumerate(ykey[0:len(iarr)]):
    ax=plt.subplot(gs[iarr[i],jarr[i]])
    ax.set_title(k, fontsize=12, y=1.01)
    rfi_flagdata = yfucngain[k].copy()
    rfi_flagdata[yflag_smooth[k]] *= np.nan
    ax.imshow(np.angle(rfi_flagdata), aspect='auto', interpolation='nearest', vmin=-np.pi, vmax=np.pi)
    
    if jarr[i] == 0:
        [t.set_fontsize(10) for t in ax.get_yticklabels()]
        ax.set_ylabel('time integrations', fontsize=10)
    if iarr[i] == 1:
        [t.set_fontsize(10) for t in ax.get_xticklabels()]
        ax.set_xlabel('freq channel', fontsize=10)

fig.savefig('/lustre/aoc/projects/hera/tbilling/polim/calibration/plots/4yPHASE_{}.png'.format(yfilename))
#plt.show()






















#%%
parser = argparse.ArgumentParser(description='Waterfall Plots of calfits files.')

parser.add_argument('Path', type= str, metavar='', requred=True, help='Path to the calfits file(e.g. /path/to/directory/)')
parser.add_argument('XX_filename', type=str, metavar='', requred=True, help='Name of the xx-polarization calfits file.')
parser.add_argument('YY_filename', type=str, metavar='', requred=True, help='Name of the yy-polarization calfits file.')
#if you want the positional argument to be optional then write '--Path'.

args = parser.parse_args()

#path = "/lustre/aoc/projects/hera/H1C_IDR2/IDR2_1/2458098/"
#ext = ".abs.calfits" #".sooth_abs.calfits"
#xxfiles = sorted(glob.glob(("{0}/zen.2458098.*.xx.HH.uv"+ext).format(args.Path)))
#yyfiles = sorted(glob.glob(("{0}/zen.2458098.*.yy.HH.uv"+ext).format(args.Path)))
#xgains, xflags = load_cal(xxfiles[i],meta=False)
#ygains, yflags = load_cal(yyfiles[i],meta=False)

if __name__ == '__main__': #This only runs is we run directly but not if you try to import it.
    import matplotlib.pyplot as plot
    import numpy as np
    
    xxfiles = sorted(glob.glob(("{0}"+args.XX_filename).format(args.Path)))
    yyfiles = sorted(glob.glob(("{0}"+args.YY_filename).format(args.Path)))
    
# If you want to look at each file and what the data does over an entire night.
    xgains_fullnight, xflags_fullnight = load_cal(xxfile,return_meta=False)
    ygains_fullnight, yflags_fullnight = load_cal(yyfile,return_meta=False)
    key = xgains.keys()
    
    ncolms = 3
    nrows = np.int(np.ceil(float(len(key))/ncolms))
    
    fig, axes = plt.subplots(nrows, ncolms, figsize=(14, 14*float(len(key))/ncolms))
    fig.subplots_adjust(wspace=0.1, hspace=0.3)
    fig.suptitle("Calfits Amplitude Waterfalls for East Pol & file = {}".format(args.XX_filename), fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    
    k = 0
    for i in range(nrows):
        for j in range(ncolms):
            ax = axes[i, j]
            key_val = key[k]
            
            rfi_flagdata = xgains[key_val].copy()
            rfi_flagdata[xflags[key_val]] *= np.nan
            ax.imshow(np.abs(rfi_flagdata), aspect='auto', interpolation='nearest')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_title("{}", fontsize=12, y=1.01)

    
    
    
#----------------------------------------------------------
# If you want to look at each file and what the data does over 60 time integrations (I still can't get it to work but....)
#----------------------------------------------------------
    for xxfile in xxfiles:
        xgains, xflags = load_cal(xxfile,return_meta=False)
        
        key = xgains.keys()

        ncolms = 3
        nrows = int(len(key)/ncolms)

        fig, axes = plt.subplots(nrows, ncolms, figsize=(14, 14*float(len(key)/ncolms)))
        fig.subplots_adjust(wspace=0.1, hspace=0.3)
        fig.suptitle("Calfits Amplitude Waterfalls for East Pol & file = {}".format(args.XX_filename), fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        k = 0
        for i in range(nrows):
            for j in range(ncolms):
                ax = axes[i, j]
                key_val = key[k]
                
                rfi_flagdata = xgains[key_val].copy()
                rfi_flagdata[xflags[key_val]] *= np.nan
                ax.imshow(np.abs(rfi_flagdata), aspect='auto', interpolation='nearest', vmax=0.06, vmin=np.abs(rfi_flagdata]).min())
                ax.xaxis.set_ticks_position('bottom')
                ax.set_title("{}", fontsize=12, y=1.01)
                #else:
                #    ax.axis('off')
                if j != 0:
                    ax.set_yticklabels([])
                else:
                    [t.set_fontsize(10) for t in ax.get_yticklabels()]
                    ax.set_ylabel('time integrations', fontsize=10)
                if i != nrows-1:
                    ax.set_xticklabels([])
                else:
                    [t.set_fontsize(10) for t in ax.get_xticklabels()]
                    ax.set_xlabel('freq channel', fontsize=10)

                k += 1
    fig.savefig(args.XX_filename+'.png')



