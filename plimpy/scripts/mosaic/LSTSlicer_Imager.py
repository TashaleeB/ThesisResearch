from __future__ import print_function, division, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import argparse,os,time,sys

from IPython.display import clear_output
from pyuvdata import UVData
from astropy.io import fits
from astropy.coordinates import Angle
from astropy import units as u

sys.path.insert(0,'/lustre/aoc/projects/hera/tbilling/')
import lst_slicer # python script that usually need
#from ja import Timer

"""
    This was based on https://github.com/UPennEoR/plimpy/blob/master/PolarizedMosaics/LST_Slicer.ipynb
"""

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

def zenstem(filelist):
    stemlist = []
    for f in filelist:
        tmp = (os.path.basename(f)).split('.')
        stemlist.append(tmp[0]+'.'+tmp[1]+'.'+tmp[2])
    stemlist.sort()
    return stemlist

# Define Arguments
a = argparse.ArgumentParser(description="Run using python as: python LSTSlicer_Imager.py <args>")
a.add_argument('--start_jd', '-sjd', type=str, help='Starting JD value eg. "2458099".',required=True)
a.add_argument('--file_path', '-fp', type=str, help='Path to file. eg. "/lustre/aoc/projects/hera/jaguirre/PolarizedMosaics/"',required=True)
a.add_argument('--image_path', '-idf', type=str, help='Path to the where you would like to sace your image.eg. "/lustre/aoc/projects/hera/jaguirre/PolarizedMosaics/"',
               required=True)
a.add_argument('--outdata_path', '-odf', type=str, help='Output data path.eg. "/lustre/aoc/projects/hera/jaguirre/PolarizedMosaics/"',
               required=True)

if __name__ == '__main__':

    args = a.parse_args()
    
    radDay = 2*np.pi
    interval = 247.5645/(24*3600.)*2*np.pi # A magic number

    JD = args.start_jd #'2458099'
    #interval = 2./(24*60.)*2*np.pi # 2 minutes
    path = args.file_path #'/lustre/aoc/projects/hera/jaguirre/PolarizedMosaics/'+JD+'/'
    imaged_path = args.image_path #'/lustre/aoc/projects/hera/tbilling/'
    outpath = args.outdata_path #'/lustre/aoc/projects/hera/tbilling/'

    sliced, master_dict = lst_slicer.slice_day (path, JD, interval)

    lsts = []
    jds = []
    filenames = []

    for key in list(master_dict.keys()):
        jds.append(key[0])
        lsts.append(key[1])
        filenames.append(master_dict[key])
    srt = np.argsort(jds)
    lsts = np.array(lsts)[srt]
    jds = np.array(jds)[srt]
    filenames = np.array(filenames)[srt]

    def lststr(angle):
        h, m, s = Angle(angle).hms
        #print(h, m, s)
        return str('%02.0f' % h) + str('%02.0f' % m) + str('%02.0f' % s)

    dec0 = -30.8 # degree
    fornax = np.array([[(3+24/60.)/24.*360, -(37+16/60.), 260],
                       [(3+(21+40/60.)/60.)/24.*360, -(37+10/60.), 490],
                       [(3+(22+43/60.)/60.)/24.*360, -(37+(12+2/60.)/60.), 2]])

    nsamps = []
    lst_grid = []

    for i in np.arange(0, radDay, interval):
        wh = np.logical_and(lsts >= i, lsts < i+interval)
        nsamp = wh.sum()
        if nsamp > 0:
            if nsamp >= 23: # Another magic number
                necessary_files = np.unique(filenames[wh])
                print(i*24/(2.*np.pi), nsamp, necessary_files)
                nsamps.append(nsamp)
                lst_grid.append(i)
    
                uvd = UVData()
                print("Created UVdata object",time.asctime(time.localtime(time.time())))
                uvd.read(list(necessary_files))
                print("Finised reading in files",time.asctime(time.localtime(time.time())))

                uvd.select(times=jds[wh])
                print("Selected Data",time.asctime(time.localtime(time.time())))
                outfile = 'zen.'+lststr(i*u.rad)+'_'+lststr((i+interval)*u.rad)+'.calibrated.HH.uvfits'
                print("Sliced File Name",outfile)
                uvd.write_uvfits(outpath+outfile, spoof_nonessential=True,force_phase=True)
                print("Finished writing uvfits file",time.asctime(time.localtime(time.time())))
                
                # Create CLEAN Mask
               
                ra0 = i # ra [rad] that the lst sliced file is phased to.
                data_hdul = fits.open('/lustre/aoc/projects/hera/aseidel/asu.fit')
                data = data_hdul[2].data
                #in_bounds = data[lst_slicer.within_bounds(data['Fintwide'], np.deg2rad(data['RAJ2000']), np.deg2rad(data['DEJ2000']), np.deg2rad(ra0), np.deg2rad(dec0), np.deg2rad(10.), 1)]
                in_bounds = data[lst_slicer.within_bounds(data['Fintwide'], np.deg2rad(data['RAJ2000']), np.deg2rad(data['DEJ2000']), ra0, np.deg2rad(dec0), np.deg2rad(10.), 1)]
                
                #fornax_inb = fornax[lst_slicer.within_bounds(fornax[:,2], np.deg2rad(fornax[:,0]), np.deg2rad(fornax[:,1]), np.deg2rad(ra0), np.deg2rad(dec0), np.deg2rad(30.), 0)]
                fornax_inb = fornax[lst_slicer.within_bounds(fornax[:,2], np.deg2rad(fornax[:,0]), np.deg2rad(fornax[:,1]), ra0, np.deg2rad(dec0), np.deg2rad(30.), 0)]
                print("Solved for boundary.",time.asctime(time.localtime(time.time())))
                
                mask_file_name = outpath+'zen.'+lststr(ra0*u.rad)+'_'+lststr((ra0+interval)*u.rad)
                mask_file = open(mask_file_name+".masks.txt", "w")
                mask_file.write("#CRTFv0\n")
                mask_file.writelines(["circle[[" + str(x[0]) + "deg, " + str(x[1]) + "deg], 0.5deg]\n" for x in in_bounds])
                mask_file.writelines(["circle[[" + str(x[0]) + "deg, " + str(x[1]) + "deg], 0.5deg]\n" for x in fornax_inb])
                mask_file.close()
                print("Mask txt file created.",time.asctime(time.localtime(time.time())))
        
                # Create Text file with correct phase_center per LST
                #timer.start()
                LST_phscenter_file = mask_file_name
                LST_phscenter = open(LST_phscenter_file+".phase_center_per_LST_slice.txt", "w")
                LST_phscenter.write(outpath+outfile + "\n") # Name of the LST sliced uvfits file it's associated with.
                LST_phscenter.write(str(ra0) + "\n")
                LST_phscenter.write(str(dec0) + "\n")
                LST_phscenter.close()
                print("Phase Center for LST txt file created.",time.asctime(time.localtime(time.time())))

                # Make MS and CASA Dirty and Deconvolved Image
                print("Starting CASA...",time.asctime(time.localtime(time.time())))
                path_file = mask_file_name
                os.system("casa -c /lustre/aoc/projects/hera/tbilling/CASA_imager.py -fp "+ path_file)
                print("MS and Images are made.",time.asctime(time.localtime(time.time())))

