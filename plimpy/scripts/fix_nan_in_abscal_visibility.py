#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 UPennEoR
# Licensed under the 2-clause BSD License

"""
   The purpose of this script is to read in a uvdata object and if there are any 'NAN' values it will convert them to zero values. Then it converts the the file to a uvfits files.
    
"""

from __future__ import print_function, division, absolute_import
from pyuvdata import UVData, UVCal
import hera_cal
import numpy as np
import argparse
import os,h5py,copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ap = argparse.ArgumentParser(description="Convert CASA gain solutions in hdf5 files to .calfits files")

ap.add_argument("--uv_file", type=str, help="name of uvfile to apply calibration to", required=True)

def main(ap):
    # read in UVData object
    uv_file = args.uv_file
    uvd = UVData()
    uvd.read(uv_file)

    # Determine Whether NAN exisits or not
    nrows,ncolms = np.argwhere(np.isnan(uv.data_array)).shape
    lists = np.argwhere(np.isnan(uv.data_array))
    
    if nrows !=0 or ncolms !=0:
        uv.data_array=np.nan_to_num(uv.data_array)
        uv.write_uvfits(uvfile+'.nancor.uvfits',spoof_nonessential=True,force_phase=True)
    
    else:
        print("No NAN")
        continue


if __name__ == '__main__':
    # parse args
    args = ap.parse_args()

    main(ap)
