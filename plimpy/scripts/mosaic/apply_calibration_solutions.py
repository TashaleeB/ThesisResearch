from __future__ import print_function, division, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import argparse,os,glob

from IPython.display import clear_output
from pyuvdata import UVData, UVCal, utils
from copy import deepcopy


def progress_bar(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    
    block = int(round(bar_length * progress))
    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)

# Define Arguments
a = argparse.ArgumentParser(description="Run using python as: python apply_calibration_solutions.py <args>")
a.add_argument('--file_path', '-fp', type=str, help='Path to file. eg. "/lustre/aoc/projects/hera/jaguirre/PolarizedMosaics/"',required=True)
a.add_argument('--rawdata_path', '-rdf', type=str, help='Path to the raw data.eg. "/lustre/aoc/projects/hera/jaguirre/PolarizedMosaics/"',
               required=True)
a.add_argument('--outdata_path', '-odf', type=str, help='Output data path.eg. "/lustre/aoc/projects/hera/jaguirre/PolarizedMosaics/"',
               required=True)

if __name__ == '__main__':
    
    args = a.parse_args()

    path = args.file_path #'/lustre/aoc/projects/hera/H1C_IDR2/IDR2_2/2458098/'
    rawpath = args.rawdata_path #'/lustre/aoc/projects/hera/H1C_IDR2/2458098/'
    cal_path = path # '/lustre/aoc/projects/hera/H1C_IDR2/IDR2_2/2458098/'
    out_path = args.outdata_path #'/lustre/aoc/projects/hera/tbilling/{}/'

    # Get the files which satisfy the sun cut
    imagefits = glob.glob(path+'zen.*.HH.calibrated.uvh5_image/*.image.image.fits')
    uvh5stem = []
    uvh5files = []
    for i,imf in enumerate(imagefits):
        tmp = os.path.split(imf)[-1].split('.')
        uvh5stem.append(tmp[0]+'.'+tmp[1]+'.'+tmp[2])
        uvh5files.append(rawpath+uvh5stem[i]+'.HH.uvh5')
    uvh5stem.sort()
    uvh5files.sort()

    for i,rf in enumerate(uvh5files[0:5]):
        progress_bar(i/len(uvh5files))
        
        raw_file = rf
        cal_file = cal_path+uvh5stem[i]+'.HH.smooth_abs.calfits'
        out_file_stem = out_path+uvh5stem[i]+'.HH.calibrated'
        print(raw_file)
        print(cal_file)
        print(out_file_stem)
        
        # Read the data
        uvd = UVData()
        uvd.read(raw_file)
        
        uvc = UVCal()
        uvc.read_calfits(cal_file)
        
        uv_calibrated = utils.uvcalibrate(uvd, uvc, inplace=False)
        
        uv_calibrated.write_uvh5(out_file_stem+'.uvh5', clobber = True)
        
        uv_calibrated.write_uvfits(out_file_stem+'.uvfits',force_phase = True, spoof_nonessential = True)

