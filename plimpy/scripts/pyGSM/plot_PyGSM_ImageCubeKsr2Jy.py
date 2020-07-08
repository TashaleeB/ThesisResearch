#!/bin/sh

#  Model_from_PyGSM_image_cube.py
#  
#
#  
"""
   This take the PyGSM image cube you made using Model_from_PyGSM_image_cube.py that has units of K*sr and converts it to Jy.
"""
from __future__ import print_function, division, absolute_import
import os, glob, sys
import numpy as np
import matplotlib.pyplot as plt

from pyuvdata import UVData
from astropy.io import fits
from astropy import units
from astropy import constants as const


filename = 'GSM_Model_of_GC_Jy_1536_WithHeaderCopiedToIt.fits'
uvdataobject = 'zen.2458560.69342.HH.uvh5'

uvd = UVData()
uvd.read_uvh5(uvdataobject)

fitsobject = fits.open(filename)
freqs = np.linspace(48.8,232.4,1024,endpoint=False)

#The maps below 20 GHz are in Rayleigh-Jeans temperature and using eq #9 in this
# paper, https://arxiv.org/pdf/1605.04920.pdf , this is how we convert from K*sr to MJy
#It is independent of the beam, because it is applied as
# a conversion from a sky model in Kelvin to one in Jy/sr, and can be pulled outside the
# visibility integral.
wvl = (const.c / (freqs*units.Hz)).to(units.m).value
Ksr2Jy = np.reshape(np.outer(np.ones(((mjysr[0].data).shape[3])**2), 1.e26 * 2. * const.k_B.value/np.power(wvl,2).squeeze()), (mjysr[0].data[0]).shape)

fitsobject[0].data[0] *= Ksr2Jy
gsm_model_name_Jyfits = 'GSM_Model_Jy_1536_imagecube.fits'
fitsobject.writeto("{}".format(gsm_model_name_Jyfits),overwrite=True)

