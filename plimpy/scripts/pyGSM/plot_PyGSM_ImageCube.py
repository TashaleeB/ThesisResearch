#!/bin/sh

#  Model_from_PyGSM_image_cube.py
#  
#
#  
"""
   Use this script to generate an model image cube that CASA can use to calibrate to. Find the pixel associated with your flux source, feed pygsm a range of frequencies and then create, convert from [K Sr] to [Jy], then save it to a FITS file. From there we apply the primary beam to the model to create a "preceived flux model", lastly we export this new file to the proper CASA format. For the fitsboject you can use any fits file with a proper header. You will also need a .npz file called "freq_info.npz" that contains the freq intervals. You could just get this information from any MS file. This is what I did and then I wrote it out to an .npz file.
"""
import os, glob, sys
import numpy as np
import healpy as hp
import pygsm
import matplotlib.pyplot as plt

from astropy_healpix import HEALPix
from astropy import wcs
from astropy.io import fits
from astropy.coordinates import SkyCoord, Galactic

from astropy import units as u
from astropy import constants as c

#name of the fits file to take HEADER from. You can use any file
fitsobject = 'mult_spec_chan_1024_gc.2457548.uvcRP.abscal.image.fits'

# Build freq. range. You can get this information within any MS file.
f=np.load('SPECTRAL_WINDOW.npz')
nchan = f['NUM_CHAN'][0] #(1,)
df = f['CHAN_WIDTH'][0] #(1024,1) was 0.097751711 MHz but now its 97656.25 Hz
freqs = f['CHAN_FREQ']*1e-6 #(1024,1) convert to [MHz] because it's originally in [Hz]

nstokes = 4 # Number of Stokes Parameters
npix = 512


#pixsize = 500./3600. # I don't think I've ever used this in this script.

# Load the FITS hdulits
data, header = fits.getdata(fitsobject, header=True)

# Parse the WCS keywords in the primary HDU
w = wcs.WCS(header)

# Create grid and convert to Equatorial Coordinates
lon, lat = np.meshgrid(np.arange(1,npix+1),np.arange(1,npix+1))
ra, dec, dummy_freq, dummy_stokes = w.all_pix2world(lon,lat,1,1,1) #Right ascension and declination as seen on the inside of the celestial sphere
c_icrs = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs') # Coordinates Object

# Transform to Galactic Coordinates
c_galactic = c_icrs.galactic

# Conversion to healpix pixels
nside=1024 # for GSM2008 512 for GSM2016 1024
hp_to_cube = hp.ang2pix(nside,np.pi/2.-c_galactic.b.rad,c_galactic.l.rad)

# ----------------------------
#   2016 Version Corrections
# ----------------------------

# Generate Simulation of the sky
cube = np.zeros([nstokes,nchan,npix,npix])
gsm_2016 = pygsm.GlobalSkyModel2016(freq_unit='MHz', unit='MJysr', resolution='hi',
                                    theta_rot=0, phi_rot=0) #units K*Steradian
for i,freq in enumerate(freqs):
    print i
    gsm_2016.generate(freq)
    this_hp_map = gsm_2016.generated_map_data
    this_gc_map = this_hp_map[hp_to_cube]
    cube[0,i,:,:] = this_gc_map

# Pick a channel and plot it [Stokes, Freq, RA, Dec]
plt.figure(figsize=(10,10))
plt.imshow(np.log10(cube[0,5,:,:]),aspect='auto')
plt.colorbar()
plt.axis('equal')
plt.xlabel('nPixel')
plt.ylabel('nPixel')
#plt.savefig("")
plt.show()

#PrimaryHDU
hdu = fits.PrimaryHDU(data=cube,header=header)
for k in header.keys():
    if k == '':
        continue
    if k == 'HISTORY':
        continue
    hdu.header.append((k,header[k]))

# write new GSM model out to a FITS file
hdu.writeto('GSM_Model_of_GC_MJy_1024StokesIQUV_WithHeaderCopiedToIt.fits',overwrite=True)

# Remember, the flux density is in [MJy] so we need to read in the FITS file and convert it to [Jy]
mjysr = fits.open('GSM_Model_of_GC_MJy_1024StokesIQUV_WithHeaderCopiedToIt.fits.fits')
mjysr[0].data *=1e6
mjysr[0].header['BUNIT']='Jy/pixel' # Change the units
mjysr.writeto('GSM_Model_of_GC_Jy_1024StokesIQUV_WithHeaderCopiedToIt.fits',overwrite=True)

# Convert FITS file to CASA image file
fits2CASA_command = "casa --nologger --nocrashreport --nogui --agg -c 'importfits('GSM_Model_of_GC_Jy_1024StokesIQUV_WithHeaderCopiedToIt.fits', 'GSM_Model_of_GC_Jy_1024StokesIQUV_WithHeaderCopiedToIt.image', overwrite=True)'"
os.system(fits2CASA_command)
#/lustre/aoc/projects/hera/tbilling/polim/calibration/old_GSM_Model_of_GC_Jy_1024StokesIQUV_WithHeaderCopiedToIt.fits
imtrans(imagename="GSM_Model_of_GC_Jy_1024StokesIQUV_WithHeaderCopiedToIt.image",
        outfile="AxisSwap_GSM_Model_of_GC_Jy_1024StokesIQUV_WithHeaderCopiedToIt.image",order ="0132")
