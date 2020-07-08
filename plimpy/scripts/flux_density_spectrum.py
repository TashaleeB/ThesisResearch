#!/bin/sh

#  flux_density_spectrum.py
#  
#
#  Created by Tashalee Billings on 7/25/18.
#  

"""
   The purpose of this document is to look at how the flux density of the visibility data behaves at different channels. This is to be used after you make a fits image file using 'mult_spect_chan_image_CASA.py'. These values may or may not have been scaled but a good wat to check if scaling is needed is to look at this plot. This is to be ran in Python.
"""

import numpy as np, matplotlib.pyplot as plt
import glob
import argparse

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename

#realdata = fits.open("")
#simdata = fits.open("")
#realdata[0].data =*2148
#simdata[0].data =*3007

#realdata.writeto('realnewtable.fits')
#simdata.writeto('simnewtable.fits')

parser = argparse.ArgumentParser(description='Creates Unscaled Flux Density Spectrum. Run with Python as: python flux_desntiy_spectrum.py --<args>=')
parser.add_argument("--fits_image", type=str, default=None, help="Fits image file name. eg 'mult_spec_chan_zen.2457755.89889.HH.uv.TrueVis.image.fits' ")
parser.add_argument("--ra_pixel", type=int, default=None, help="Open CASA image file, pick a channel and identify the RA pixel number associated with the largest Flux Density of the GC.")
parser.add_argument("--dec_pixel", type=int, default=None, help="Open CASA image file, pick a channel and identify the DEC pixel number associated with the largest Flux Density of the GC.")

args = parser.parse_args()

if __name__ == '__main__': #This only runs is we run directly but not if you try to import it.

    filename = args.fits_image
    fitsdata=fits.open(filename)[0].data # (4, 150, 512, 512)=(stokes,nimage,RA,DEC)

    nimage = np.shape(fitsdata)[1]
    freq = np.linspace(115,188,nimage) #(start freq, end freq, num of points between)
    ra_pixel,dec_pixel = args.ra_pixel,args.dec_pixel # SIM 268,261 or # REAL 268,251 # Pixel number for GC
    
    plt.figure(figsize=(15,8))

    for ind in range(nimage):
        
        plt.plot(freq[ind],fitsdata.data[0,ind,ra_pixel,dec_pixel],'.')
        plt.xlabel("Freq [Hz]")
        plt.ylabel("Flux Density ")

    plt.title('{} Flux Density Spectrum'.format(filename))
    plt.savefig('{}_FluxDensitySpectrum.png'.format(filename))
    #plt.show()
    plt.close()

