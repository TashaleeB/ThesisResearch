#!/bin/sh

#  100_200MHz_Beam_and_GSM_XX_model.py
#  
#
#  Created by Tashalee Billings on 12/3/18.
#  
"""
   This script creates .beamfits using Nick Fagnoni CST E-Field Patterns found in his github repo.
   
   https://github.com/Nicolas-Fagnoni/Simulations/tree/master/Radiation%20patterns/E-field%20pattern%20-%20Rigging%20height%204.9%20m
   
   Then saves it as a NF_HERA_Beam_100_200MHz_xPol.beamfits.
   
   Then apply this primary beam to the model image to create a "preceived flux model" and write it out as a primary beam corrected fits file.
"""
import numpy as np
import astropy.io.fits as fits
from astropy import wcs
from pyuvdata import UVBeam
from hera_pspec.data import DATA_PATH
import os
import sys
import glob
import argparse
import shutil
import copy
import healpy
import scipy.stats as stats
from scipy import interpolate
from astropy.time import Time
from astropy import coordinates as crd
from astropy import units as u

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Build .beamfits file for XX between 100-200 MHz. The beam information can be found https://github.com/Nicolas-Fagnoni/Simulations/tree/master/Radiation%20patterns/E-field%20pattern%20-%20Rigging%20height%204.9%20m

path_to_Efield_pattern = '/lustre/aoc/projects/hera/tbilling/polim/NicolasFagnoniBeams/'
CST100_150_list = sorted(glob.glob(path_to_Efield_pattern+'HERA_4.9m_E-pattern_100-150MHz/HERA_4.9m_E-pattern_ *MHz.txt'))
CST151_200_list = sorted(glob.glob(path_to_Efield_pattern+'HERA_4.9m_E-pattern_151-200MHz/HERA_4.9m_E-pattern_ *MHz.txt'))


# I'm only passing 100 files that will be combined across the frequency axis from 100-200 MHz
CST = CST100_150_list + CST151_200_list

# Must specify telescope spec and polarization of the beam.
beam = UVBeam()
freq = [int(round(f)) for f in np.linspace(100,200,101)*1e6] # needs to be an integer
beam.read_cst_beam(CST, beam_type='power', frequency= freq,
                   feed_pol='x', rotate_pol=True, telescope_name='HERA',
                   feed_name='PAPER_dipole', feed_version='0.1',
                   model_name='E-field pattern - Rigging height 4.9m',
                   model_version='1.0')

#beam.data_array.shape #(Naxes_vec =1, Nspws=1, Npols=2, Nfreqs=2, Naxes2=181, Naxes1=360)

# Now specify the interpolation function to use.
beam.interpolation_function = 'az_za_simple'
beam.to_healpix()
beam.peak_normalize()
"""
# Doesn't work the way I thought it would work. Needs to have certain dimension and (1, 1, 2, 1024, 181, 360) is not the correct one.
# Interpolate beam to a given range of frequency values.
freq = np.linspace(100,200,1024)*1e6

interpolated_beam_values = beam.interp(freq_array=freq)
beam.data_array = interpolated_beam_values[0] # Replace
beam.data_array.shape #(Naxes_vec =1, Nspws=1, Npols=2, Nfreqs=1024, Naxes2=181, Naxes1=360)
"""
new_beam_name = 'NF_HERA_Peak_Norm_Beam_100_200MHz_xPol.beamfits''
beam.write_beamfits(new_beam_name, clobber=True)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Influenced by Nick Kern's pb.corr.py.
# Combine .beamfits file with GSM_xx_model_WHATEVER_THE_NAME_IS.fits to get perceived model.

args = argparse.ArgumentParser(description="Primary beam correction on FITS image files, given primary beam model")

args.add_argument("fitsfiles", type=str, nargs='*', help='path of image FITS file(s) to PB correct')

# PB args
args.add_argument("--multiply", default=False, action='store_true', help='multiply data by primary beam, rather than divide')
args.add_argument("--lon", default=21.42830, type=float, help="longitude of observer in degrees east")
args.add_argument("--lat", default=-30.72152, type=float, help="latitude of observer in degrees north")
args.add_argument("--time", type=float, help='time of middle of observation in Julian Date')

# HEALpix Beam args
#args.add_argument("--beamfile", type=str, help="path to primary beam healpix map in pyuvdata.UVBeam format")
#args.add_argument("--pols", type=int, nargs='*', default=None, help="Polarization integer of healpix maps to use for beam models. Default is to use polarization in fits HEADER.")

# Gaussian Beam args
args.add_argument("--ew_sig", type=float, default=None, nargs='*',
                  help="if no healpix map provided, array of gaussian beam sigmas (per freq) in the east-west direction")
args.add_argument("--ns_sig", type=float, default=None, nargs='*',
                  help="if no healpix map provided, array of gaussian beam sigmas (per freq) in the north-south direction")
args.add_argument("--gauss_freqs", type=float, default=None, nargs='*',
                  help="if no healpix map provided, array of frequencies (Hz) matching length of ew_sig and ns_sig")

# IO args
args.add_argument("--ext", type=str, default="", help='Extension prefix for output file.')
args.add_argument("--outdir", type=str, default=None, help="output directory, default is path to fitsfile")
args.add_argument("--overwrite", default=False, action='store_true', help='overwrite output files')
args.add_argument("--silence", default=False, action='store_true', help='silence output to stdout')
args.add_argument("--spec_cube", default=False, action='store_true', help='assume all fitsfiles are identical except they each have a single but different frequency')

if __name__ == "__main__":
    
    # parse args
    a = args.parse_args()
    
    beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
    
    # load pb
    if a.beamfile is not None:
         print("...loading beamfile {}".format(a.beamfile))
        # load beam
        uvb = UVBeam()
        uvb.read_beamfits(a.beamfile)
        # get beam models and beam parameters
        beam_maps = np.abs(uvb.data_array[0, 0, :, :, :])
        beam_freqs = uvb.freq_array.squeeze() / 1e6
        Nbeam_freqs = len(beam_freqs)
        beam_nside = healpy.npix2nside(beam_maps.shape[2])
        pols = uvb.polarization_array[0]
        
        # construct beam interpolation function
        def beam_interp_func(theta, phi):
            # convert to radians
            theta = copy.copy(theta) * np.pi / 180.0
            phi = copy.copy(phi) * np.pi / 180.0
            shape = theta.shape
            # loop over freq, then pol
            beam_interp = [[healpy.get_interp_val(m, theta.ravel(), phi.ravel(), lonlat=False).reshape(shape) for m in maps] for maps in beam_maps]
            return np.array(beam_interp)

    # iterate over model FITS files to be pb corrected
    for i, ffile in enumerate(a.fitsfiles):
        
        # create output filename
        if a.outdir is None:
            output_dir = os.path.dirname(ffile)
        else:
            output_dir = a.outdir

        output_fname = os.path.basename(ffile)
        output_fname = os.path.splitext(output_fname)
        if a.ext is not None:
            output_fname = output_fname[0] + '.pbcorr{}'.format(a.ext) + output_fname[1]
        else:
            output_fname = output_fname[0] + '.pbcorr' + output_fname[1]
        output_fname = os.path.join(output_dir, output_fname)
            
        # check for overwrite
        if os.path.exists(output_fname) and a.overwrite is False:
            raise IOError("{} exists, not overwriting".format(output_fname))
        
        # load hdu and get header and data
        print("...loading {}".format(ffile))
        data, header = fits.getdata(ffile, header=True)

        # determine if freq precedes stokes in header
        if head['CTYPE3'] == 'FREQ':
            freq_ax = 3
            stok_ax = 4
        else:
            freq_ax = 4
            stok_ax = 3

        # get axes info
        npix1 = head["NAXIS1"]
        npix2 = head["NAXIS2"]
        nstok = head["NAXIS{}".format(stok_ax)]
        nfreq = head["NAXIS{}".format(freq_ax)]
        
        # get polarization info
        pol_arr = np.asarray(head["CRVAL{}".format(stok_ax)] + np.arange(nstok) * head["CDELT{}".format(stok_ax)], dtype=np.int)
            
        # replace with forced polarization if provided
        if pols is not None:
            pol_arr = np.asarray([pols], dtype=np.int)

        # set beam maps
        beam_pols = uvb.polarization_array.tolist()
        beam_maps = np.array([beam_maps[beam_pols.index(p)] for p in pol_arr])
            
        # make sure required pols exist in maps
        if not np.all([p in uvb.polarization_array for p in pol_arr]):
            raise ValueError("Required polarizationns {} not found in Beam polarization array".format(pol_arr))

        # get WCS
        w = wcs.WCS(ffile)

        # convert pixel to equatorial coordinates
        lon_arr, lat_arr = np.meshgrid(np.arange(npix1), np.arange(npix2))
        lon, lat, s, f = w.all_pix2world(lon_arr.ravel(), lat_arr.ravel(), 0, 0, 0)
        lon = lon.reshape(npix2, npix1)
        lat = lat.reshape(npix2, npix1)

        # convert from equatorial to spherical coordinates
        loc = crd.EarthLocation(lat=a.lat*u.degree, lon=a.lon*u.degree)
        """
        # How I find the middle time within the data file
        uvd = UVDdata()
        uvd.read_uvh5("zen.2457548.46619.xx.HH.uvcRPCS.uvh5")
        jd_array = np.unique(uvd.time_array)
        mid_time_index = int(np.round(jd_array.shape[0]/2.))
        jd_array[mid_time_index] # The one I found happen to be 2457548.469734946
        """
        time = Time(a.time, format='jd', scale='utc')
        equatorial = crd.SkyCoord(ra=lon*u.degree, dec=lat*u.degree, frame='fk5', location=loc, obstime=time)
        altaz = equatorial.transform_to('altaz')
        theta = np.abs(altaz.alt.value - 90.0)
        phi = altaz.az.value
            
        # get data frequencies
        if freq_ax == 3:
            data_freqs = w.all_pix2world(0, 0, np.arange(nfreq), 0, 0)[2] / 1e6
        else:
            data_freqs = w.all_pix2world(0, 0, 0, np.arange(nfreq), 0)[3] / 1e6
            Ndata_freqs = len(data_freqs)
            
        if i == 0 or a.spec_cube is False:
            # evaluate primary beam
            print("...evaluating PB")
            pb = beam_interp_func(theta, phi)

        # interpolate primary beam frequencies onto data frequencies
        print("...interpolating PB")
        pb_shape = (pb.shape[1], pb.shape[2])
        pb_interp = interpolate.interp1d(beam_freqs, pb, axis=1, kind='linear', fill_value='extrapolate')(data_freqs)

        # data shape is [naxis4, naxis3, naxis2, naxis1]
        if freq_ax == 4:
            pb_interp = np.moveaxis(pb_interp, 0, 1)
            
        # divide or multiply by primary beam
        if a.multiply is True:
            print("...multiplying PB into image")
            data_pbcorr = data * pb_interp
        else:
            print("...dividing PB into image")
            data_pbcorr = data / pb_interp
            
        # change polarization to interpolated beam pols
        head["CRVAL{}".format(stok_ax)] = pol_arr[0]
        if len(pol_arr) == 1:
            step = 1
        else:
            step = np.diff(pol_arr)[0]

        head["CDELT{}".format(stok_ax)] = step
        head["NAXIS{}".format(stok_ax)] = len(pol_arr)

        print("...saving {}".format(output_fname))
        fits.writeto(output_fname, data_pbcorr, head, overwrite=True)

        output_pb = output_fname.replace(".pbcorr.", ".pb.")
        print("...saving {}".format(output_pb))
        fits.writeto(output_pb, pb_interp, head, overwrite=True)

"""
    # Commands used
    
python 100_200MHz_Beam_and_GSM_XX_model.py  --outdir ./ \
                                            --pol -5 \
                                            --time 2457548.469734946 \
                                            --ext pbcorr \
                                            --lon 21.42830 \
                                            --lat -30.72152 \
                                            --overwrite \
                                            --multiply \
                                            GSM_Model_of_GC_Jy_1024StokesI_WithHeaderCopiedToIt.fits
"""


"""
import numpy as np
import astropy.io.fits as fits
from astropy import wcs
from pyuvdata import UVBeam
from hera_pspec.data import DATA_PATH
import os
import sys
import glob
import argparse
import shutil
import copy
import healpy
import scipy.stats as stats
from scipy import interpolate
from astropy.time import Time
from astropy import coordinates as crd
from astropy import units as u

beamfile = "HERA_NF_dipole_power.beamfits"
i==0

# load pb
if beamfile is not None:
# load beam
uvb = UVBeam()
uvb.read_beamfits(beamfile)
# get beam models and beam parameters
beam_maps = np.abs(uvb.data_array[0, 0, :, :, :])
beam_freqs = uvb.freq_array.squeeze() / 1e6
Nbeam_freqs = len(beam_freqs)
beam_nside = healpy.npix2nside(beam_maps.shape[2])
pols = uvb.polarization_array[0]

# construct beam interpolation function
def beam_interp_func(theta, phi):
# convert to radians
theta = copy.copy(theta) * np.pi / 180.0
phi = copy.copy(phi) * np.pi / 180.0
shape = theta.shape
# loop over freq, then pol
beam_interp = [[healpy.get_interp_val(m, theta.ravel(), phi.ravel(), lonlat=False).reshape(shape) for m in maps] for maps in beam_maps]
return np.array(beam_interp)
ffile = 'GSM_Model_of_GC_Jy_1024StokesI_WithHeaderCopiedToIt.fit'
output_fname = os.path.basename(ffile)
output_fname = os.path.splitext(output_fname)
output_fname = output_fname[0] + '_Physical_dipole_power2.pbcorr' + output_fname[1]

hdu = fits.open('GSM_Model_of_GC_Jy_1024StokesI_WithHeaderCopiedToIt.fits')

# get header and data
head = hdu[0].header
data = hdu[0].data

# determine if freq precedes stokes in header
if head['CTYPE3'] == 'FREQ':
    freq_ax = 3
    stok_ax = 4
else:
    freq_ax = 4
    stok_ax = 3

# get axes info
npix1 = head["NAXIS1"]
npix2 = head["NAXIS2"]
nstok = head["NAXIS{}".format(stok_ax)]
nfreq = head["NAXIS{}".format(freq_ax)]

# get polarization info
pol_arr = np.asarray(head["CRVAL{}".format(stok_ax)] + np.arange(nstok) * head["CDELT{}".format(stok_ax)], dtype=np.int)

# replace with forced polarization if provided
if pols is not None:
    pol_arr = np.asarray([pols], dtype=np.int)

# set beam maps
beam_pols = uvb.polarization_array.tolist()
beam_maps = np.array([beam_maps[beam_pols.index(p)] for p in pol_arr])

# get WCS
w = wcs.WCS('GSM_Model_of_GC_Jy_1024StokesI_WithHeaderCopiedToIt.fits')

# convert pixel to equatorial coordinates
lon_arr, lat_arr = np.meshgrid(np.arange(npix1), np.arange(npix2))
lon, lat, s, f = w.all_pix2world(lon_arr.ravel(), lat_arr.ravel(), 0, 0, 0)
lon = lon.reshape(npix2, npix1)
lat = lat.reshape(npix2, npix1)

# convert from equatorial to spherical coordinates
loc = crd.EarthLocation(lat=-30.72152*u.degree, lon=21.42830*u.degree)
time = Time(2457548.469734946, format='jd', scale='utc')
equatorial = crd.SkyCoord(ra=lon*u.degree, dec=lat*u.degree, frame='fk5', location=loc, obstime=time)
altaz = equatorial.transform_to('altaz')
theta = np.abs(altaz.alt.value - 90.0)
phi = altaz.az.value

# get data frequencies
if freq_ax == 3:
    data_freqs = w.all_pix2world(0, 0, np.arange(nfreq), 0, 0)[2] / 1e6
else:
    data_freqs = w.all_pix2world(0, 0, 0, np.arange(nfreq), 0)[3] / 1e6
    Ndata_freqs = len(data_freqs)


pb = beam_interp_func(theta, phi)

# interpolate primary beam frequencies onto data frequencies
pb_shape = (pb.shape[1], pb.shape[2])
pb_interp = interpolate.interp1d(beam_freqs, pb, axis=1, kind='linear', fill_value='extrapolate')(data_freqs)

# data shape is [naxis4, naxis3, naxis2, naxis1]
if freq_ax == 4:
    pb_interp = np.moveaxis(pb_interp, 0, 1)

# divide or multiply by primary beam
data_pbcorr = data * pb_interp

# change polarization to interpolated beam pols
head["CRVAL{}".format(stok_ax)] = pol_arr[0]
if len(pol_arr) == 1:
    step = 1
else:
    step = np.diff(pol_arr)[0]

head["CDELT{}".format(stok_ax)] = step
head["NAXIS{}".format(stok_ax)] = len(pol_arr)

fits.writeto(output_fname, data_pbcorr, head, overwrite=True)

output_pb = output_fname.replace(".pbcorr.", ".pb.")
fits.writeto(output_pb, pb_interp, head, overwrite=True)
"""
