#!/bin/sh

#  mult_spect_chan_image.py
#  
#
#  Created by Tashalee Billings on 7/28/18.
#  

"""
    The purpose of this document is to create the image file in CASA that contains channel incremented stokes images into one file. This will allow you to look at the Stokes Visibilties at different channels. This is to be executed using CASA.
"""

import glob
import argparse

parser = argparse.ArgumentParser(description='Creates an Image file in CASA that allows you to look at Stokes Visibilities at different channel (150 to 900 in increments of 5). Run with casa as: casa -c mult_spect_chan_image.py --<args>=')

parser.add_argument("-c", type=str, help="Calibrationg for different permutations.")
parser.add_argument("--ms", type=str, default=None, help="The name of the Measuement Set you want to use eg. 'zen.2457755.89889.conj.HH.uv.TrueVis.MS'")

args = parser.parse_args()


if __name__ == '__main__': #This only runs is we run directly but not if you try to import it.
    
    m = args.ms

    fitsname = 'mult_spec_1024_chan_'+m[:-3]+'.image.fits'
    imagename = 'mult_spec_1024_chan_'+m[:-3]+'.image'

    clean(m,imagename[:-6],niter=0,weighting = 'briggs',robust =0,imsize =[512 ,512]
          ,cell=['500 arcsec'],phasecenter = 'J2000 17h45m40.0409s -29d0m28.118s'
          ,mode='channel',nterms =1,spw='0',nchan=1024, start=0, width=1
          ,stokes='IQUV', interactive=False, npercycle=5, threshold='0.1mJy/beam')

    exportfits(imagename,fitsname) # Makes 1 image fits file containing a certain number of images you stated in nimage.