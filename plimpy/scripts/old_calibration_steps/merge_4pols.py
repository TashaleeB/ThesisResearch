#!/bin/sh

#  merge_4pols.py
#  
#
#  Created by Tashalee Billings on 5/23/18.
#  

import numpy as np
import argparse
import shutil
import os, glob, sys

from astropy import units as u
from astropy import constants as c
from pyuvdata import UVData

pols = ['xx','yy','xy','yx']

parser = argparse.ArgumentParser(description='Merges the auto/cross correlations of the MIRIAD file.')

parser.add_argument("--prefix", type=str, default=None, help="Miriad prefix, e.g. zen.2457548.46619")
parser.add_argument("--suffix", type=str, default=None, help="Miriad prefix, e.g. .HH.uv")

args = parser.parse_args()

prefix = args.prefix
suffix = args.suffix

# Merge the 4 correlations to create 4-pol image

#For Simulated Data
#----------------------
#This Needs to be done file by file then after that you can merge the 4 pol
uvd = UVData()
uvd.read_miriad('file.uv')
# Fix conjugation error
uvd.data_array = np.conj(uvd.data_array)
# Fix calibration to Jy
# Look, Ma!  No loops!
wvl = (c.c/(uvd.freq_array*u.Hz)).to(u.m).value
Ksr2Jy = np.reshape(np.outer(np.ones(uvd.Nblts),1.e26*2.*c.k_B.value/np.power(wvl,2).squeeze()),uvd.data_array.shape)
uvd.data_array *= Ksr2Jy
uvd.write_miriad('file.uv')
#----------------------


files = []
for p in pols:
    files.append(prefix+'.'+p+suffix)

uv = UVData()
print 'Reading data'
for f in files:
    print f
uv.read_miriad(files)

miriadfile = prefix+suffix
print 'Writing data to '+miriadfile+".uvfits"
uv.write_uvfits(miriadfile+".uvfits", force_phase=True, spoof_nonessential=True)

# Convert To UVFITS file
"""print("Converting Miriad to uvfits.")
os.system("python miriad2uvfits.py " + miriadfile)"""
m = glob.glob(miriadfile+".uvfits")[0]
print("You now have a uvfits file named " + m)
