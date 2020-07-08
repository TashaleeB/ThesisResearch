#!/bin/sh

#  imaging_pipeline.py
#  
#
#  Created by Tashalee Billings on 2/6/19.
#  

"""
    This purpose of this document is make a CASA image that is not deconvolved 
    python imaging_pipeline.py --beg_jd= --ext= --root_dir=
    
"""
import glob,os,sys,time
import argparse
import subprocess

from pyuvdata import UVData

#-------------------------------------------------------------------------------
# Command-Line Parameters
#-------------------------------------------------------------------------------

args =  argparse.ArgumentParser(description="imaging_pipeline.py --<args> ")

args.add_argument("--beg_jd", default=None, type=str, help='The beggining JD values excluding the last two digits. eg. "2457548.46619" would be written as "24575" ')
args.add_argument("--ext", default=None, type=str, help='Type the extension of the file. eg. "uvcRPCS.uvh5"')
args.add_argument("--root_dir",default='/data4/paper/HERA19Golden/CalibratedData/', type=str, help='write out the head directory. eg "/data4/paper/HERA19Golden/CalibratedData/"')

arg= args.parse_args()

#-------------------------------------------------------------------------------
# Parameters
#-------------------------------------------------------------------------------

pols = ['xx', 'yy', 'xy', 'yx']
search_name = arg.root_dir + arg.beg_jd + '*/zen.'+ arg.beg_jd + '*' + arg.ext
files = sorted(glob.glob(search_name))

#-------------------------------------------------------------------------------
# Check if the files are uvfits, miriad, or hdf5
#-------------------------------------------------------------------------------

# Starting with a uvfits file

if files[0][-6:] == 'uvfits' :
    #print("This is a uvfits file. Combining 4-pols, ['.xx', '.yy', '.xy', '.yx'], and converting to 'MS' file.")
    
    for i in range(0,len(files), 4):
        
        sub_files_of4 = []
        base = os.path.basename(files[i]) # get the base of every 4th file in the list.
        dirname = os.path.dirname(files[i])
        split_file = base.split('.')
        
        # reorder and make a subgroup of 4 files ['zen.*.xx.*.uvfits', 'zen.*.yy.*.uvfits', 'zen.*.xy.*.uvfits', 'zen.*.yx.*.uvfits']
        for pol in pols:
            reordered = split_file[0:3]+ [pol]+ split_file[4:7] # remember glob doesn't list the files in the proper order ['.xx', '.yy', '.xy', '.yx'].
            base1 = '.'.join(reordered)
            path_base = os.path.join(dirname,base1) # joins path with file name again
            sub_files_of4.append(path_base) #adds it to the list
        
        # read in list of uvfits files and make 1 4-pol uvfits file
        uvd = UVData()
        uvd.read_uvfits(sub_files_of4)
        uvfits_outfile = arg.root_dir+ split_file[1]+ '/'+ '.'.join(split_file[0:3]+split_file[4:7])+'.uvfits'
        ms_outfile = arg.root_dir+ split_file[1]+ '/'+ '.'.join(split_file[0:3]+split_file[4:6]+['MS'])
        uvd.write_uvfits(uvfits_outfile, force_phase=True, spoof_nonessential=True)

        # Convert from uvfits to MS
        uvfits2ms = subprocess.check_call(["casa","--nologger"," --nocrashreport"," --nogui", "--agg","-c", "importuvfits('{}', '{}')".format(uvfits_outfile, ms_outfile)])

        # Make dirty image and save it
        clean = "clean('{}','{}_noDecon',niter=0, weighting = 'briggs',robust =0, imsize =[512 ,512] ,pbcor=False, cell=['500 arcsec'] ,mode='mfs',nterms=1, spw='0:150~900', stokes='IQUV')".format(ms_outfile,ms_outfile[:-3])
        dirty_image = subprocess.check_call(["casa","-c",clean])
        casa_image = '{}_noDecon'.format(ms_outfile[:-3])+'.image'
        fits_img = '{}_noDecon'.format(ms_outfile[:-3])+'.fits'
        fits_image = subprocess.check_call(["casa","--nologger"," --nocrashreport"," --nogui", "--agg","-c","exportfits('{}','{}')".format(casa_image,fits_img)])
    del(uvd)

# Starting with a hdf5 file
if files[0][-4:] == 'uvh5' :
    #print("This is a hdf5 file. Combining 4-pols, ['.xx', '.yy', '.xy', '.yx'], and converting to 'uvfits' file and then to 'MS' file.")

    for i in range(0,len(files), 4):
        
        sub_files_of4 = []
        base = os.path.basename(files[i]) # get the base of every 4th file in the list.
        dirname = os.path.dirname(files[i])
        split_file = base.split('.')
        
        # reorder and make a subgroup of 4 files ['zen.*.xx.*.uvfits', 'zen.*.yy.*.uvfits', 'zen.*.xy.*.uvfits', 'zen.*.yx.*.uvfits']
        for pol in pols:
            reordered = split_file[0:3]+ [pol]+ split_file[4:7] # remember glob doesn't list the files in the proper order ['.xx', '.yy', '.xy', '.yx'].
            base1 = '.'.join(reordered)
            path_base = os.path.join(dirname,base1) # joins path with file name again
            sub_files_of4.append(path_base) #adds it to the list
    
        # read in list of hdf5 files and make 1 4-pol uvfits file
        uvd = UVData()
        uvd.read_uvh5(sub_files_of4)
        uvfits_outfile = arg.root_dir+ split_file[1]+ '/'+ '.'.join(split_file[0:3]+split_file[4:7])+'.uvfits'
        ms_outfile = arg.root_dir+ split_file[1]+ '/'+ '.'.join(split_file[0:3]+split_file[4:6]+['MS'])
        uvd.write_uvfits(uvfits_outfile, force_phase=True, spoof_nonessential=True)
        
        # Convert from uvfits to MS
        uvfits2ms = subprocess.check_call(["casa", "--nologger"," --nocrashreport"," --nogui", "--agg","-c", "importuvfits('{}', '{}')".format(uvfits_outfile, ms_outfile)])
        
        # Make dirty image and save it
        clean = "clean('{}','{}_noDecon',niter=0, weighting = 'briggs',robust =0, imsize =[512 ,512] ,pbcor=False, cell=['500 arcsec'] ,mode='mfs',nterms=1, spw='0:150~900', stokes='IQUV')".format(ms_outfile,ms_outfile[:-3])
        dirty_image = subprocess.check_call(["casa","-c",clean])
        casa_image = '{}_noDecon'.format(ms_outfile[:-3])+'.image'
        fits_img = '{}_noDecon'.format(ms_outfile[:-3])+'.fits'
        fits_image = subprocess.check_call(["casa","--nologger"," --nocrashreport"," --nogui", "--agg","-c","exportfits('{}','{}')".format(casa_image,fits_img)])

    del(uvd)

# Miriad file
else :
    #print("This must be a miriad file. Combining 4-pols, ['.xx', '.yy', '.xy', '.yx'], and converting to 'uvfits' file and then to 'MS' file.")

    for i in range(0,len(files), 4):
        
        sub_files_of4 = []
        base = os.path.basename(files[i]) # get the base of every 4th file in the list.
        dirname = os.path.dirname(files[i])
        split_file = base.split('.')
        
        # reorder and make a subgroup of 4 files ['zen.*.xx.*.uvfits', 'zen.*.yy.*.uvfits', 'zen.*.xy.*.uvfits', 'zen.*.yx.*.uvfits']
        for pol in pols:
            reordered = split_file[0:3]+ [pol]+ split_file[4:7] # remember glob doesn't list the files in the proper order ['.xx', '.yy', '.xy', '.yx'].
            base1 = '.'.join(reordered)
            path_base = os.path.join(dirname,base1) # joins path with file name again
            sub_files_of4.append(path_base) #adds it to the list
        
        # read in list of hdf5 files and make 1 4-pol uvfits file
        uvd = UVData()
        uvd.read_miriad(sub_files_of4)
        uvfits_outfile = arg.root_dir+ split_file[1]+ '/'+ '.'.join(split_file[0:3]+split_file[4:7])+'.uvfits'
        ms_outfile = arg.root_dir+ split_file[1]+ '/'+ '.'.join(split_file[0:3]+split_file[4:6]+['MS'])
        uvd.write_uvfits(uvfits_outfile, force_phase=True, spoof_nonessential=True)
        
        # Convert from uvfits to MS
        uvfits2ms = subprocess.check_call(["casa","--nologger"," --nocrashreport"," --nogui", "--agg","-c", "importuvfits('{}', '{}')".format(uvfits_outfile, ms_outfile)])
        
        # Make dirty image and save it
        clean = "clean('{}','{}_noDecon',niter=0, weighting = 'briggs',robust =0, imsize =[512 ,512] ,pbcor=False, cell=['500 arcsec'] ,mode='mfs',nterms=1, spw='0:150~900', stokes='IQUV')".format(ms_outfile,ms_outfile[:-3])
        dirty_image = subprocess.check_call(["casa","-c",clean])
        casa_image = '{}_noDecon'.format(ms_outfile[:-3])+'.image'
        fits_img = '{}_noDecon'.format(ms_outfile[:-3])+'.fits'
        fits_image = subprocess.check_call(["casa","--nologger"," --nocrashreport"," --nogui", "--agg","-c","exportfits('{}','{}')".format(casa_image,fits_img)])

    del(uvd)
