from __future__ import print_function, division, absolute_import
import numpy as np
import subprocess

script_dir = "/lustre/aoc/projects/hera/tbilling/"

# Flagged data and LST Plots
# Commands
print("Making Plots of LST Binned Data")
python = ['python', script_dir+'LSTBins_Flags.py']
cmd = python
status = subprocess.call(cmd)

print("Entering Loop to Calibrate, slice, and image the data.")
JDS = ['2458098']#,'2458099','2458101','2458102']
for jd in JDS:
    
    # Apply Calibration Solutions (apply_calibration_solutions.py)
    #Args
    path_file = '/lustre/aoc/projects/hera/H1C_IDR2/IDR2_2/{}/'.format(jd)
    rawpath = '/lustre/aoc/projects/hera/H1C_IDR2/{}/'.format(jd)
    out_path = '/lustre/aoc/projects/hera/tbilling/{}/'.format(jd)

    # Commands
    mkdir = subprocess.call(['mkdir',out_path])
    python = ['python', script_dir+'apply_calibration_solutions.py']
    cmd = python+ ['-fp', path_file, '-rdf', rawpath, '-odf', out_path]

    status = subprocess.call(cmd)

    # LST Slice Calibrated data and Image it (LSTSlicer_Imager.py)
    # Args
    JD = jd#'2458099'
    path = out_path
    imaged_path = out_path
    outpath = out_path

    # Commands
    python = ['python', script_dir+'LSTSlicer_Imager.py']
    cmd = python+ ['-sjd',JD , '-fp', path, '-idf', imaged_path, '-odf', outpath]

    status = subprocess.call(cmd)

"""
#
#Args
path = '/lustre/aoc/projects/hera/tbilling/'

# Commands
python = ['python', 'PolarizedMosaic.py']
cmd = python+ ['-fp', path]

status = subprocess.call(cmd)
"""



