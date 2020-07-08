#!/bin/sh

#  plot_califits.py
#  
#
#  Created by Tashalee Billings on 03/18/2019.
#

import numpy as np, healpy as hp, matplotlib.pyplot as plt
import casa_utils as utils # lives in same dir as this notebook
import source2file # lives in same dir as this notebook
import pygsm,time
import pyuvdata.utils as uvutils
import operator,subprocess,argparse
import os,sys,glob,yaml
import json,itertools,shutil

import colorcet as cc
ccc = cc.m_cyclic_grey_15_85_c0_s25
import matplotlib
import cmocean

from astropy.coordinates import EarthLocation
from mpl_toolkits.axes_grid1 import AxesGrid
from astropy.io import fits
from astropy import wcs
from pyuvdata import UVData
from datetime import datetime
from collections import OrderedDict as odict
from astropy.time import Time
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord, Galactic
from astropy import units as u
from astropy import constants as c
from copy import deepcopy

sys.path.insert(0,'/Users/tashaleebillings/')
from cst2ijones.jones_matrix_field  import InstrumentalJonesMatrixField
from cst2ijones.plotting import PlotMueller
from cst2ijones import spherical_coordinates_basis_transformation as scbt


def txtname(n):
    if n not in range(50,251):
        raise ValueError('no data at that frequency.')
    fname = '/Users/tashaleebillings/Desktop/research/data/NF_BEAM/HERA_4.9m_E-pattern_ {0}MHz.txt'.format(str(n))
    
    return fname
def StokesMatrix(n):
    if n not in [0,1,2,3]: raise Exception('Input must be an integer in [0,1,2,3]')
    
    if n == 0:
        p = np.array([[1.,0],
                      [0.,1.]])
    elif n == 1:
        p = np.array([[1.,0],
                      [0,-1.]])
    elif n == 2:
        p = np.array([[0,1.],
                      [1.,0]])
    elif n == 3:
        p = np.array([[0., -1j],
                      [1j,0]])
    return p

def S_Matrix(n):
    ones = np.ones((512,512),dtype=int)
    comps = np.ones_like(ones,dtype=np.complex64)*1j*0.5
    
    if n not in ["correlation","pseudo"]: raise Exception('Input must be an string in ["correlation","pseudo"]')
    
    if n =="correlation": # 512x512 Identity Image
        # Build Identity
        I = np.zeros((512,512,4,4),dtype=int)
        i_id = [0,1,2,3]
        
        for idx in range(len(i_id)):
            I[:,:,i_id[idx],i_id[idx]] += ones
        S_ = I
    
    if n =="pseudo":# 512x512 S image
        S_ = np.zeros((512,512,4,4),dtype=np.complex64)
        # Build S
        
        i_id = [0,0,1,1,2,2,3,3]
        j_id = [0,3,0,3,1,3,1,3]
        
        for idx in range(len(i_id)):
            if i_id[idx] == 1 and j_id[idx] == 3:
                S_[:,:,i_id[idx],j_id[idx]] -= ones
            
            if i_id[idx] == 3 and j_id[idx] == 1:
                S_[:,:,i_id[idx],j_id[idx]] -= comps
            
            if i_id[idx] == 3 and j_id[idx] == 2:
                S_[:,:,i_id[idx],j_id[idx]] += comps
            
            if i_id[idx] == 0 and j_id[idx] == 0:
                S_[:,:,i_id[idx],j_id[idx]] += ones
            
            if i_id[idx] == 0 and j_id[idx] == 3:
                S_[:,:,i_id[idx],j_id[idx]] += ones
            
            if i_id[idx] == 1 and j_id[idx] == 0:
                S_[:,:,i_id[idx],j_id[idx]] += ones
            
            if i_id[idx] == 2 and j_id[idx] == 1:
                S_[:,:,i_id[idx],j_id[idx]] += ones
            
            if i_id[idx] == 2 and j_id[idx] == 2:
                S_[:,:,i_id[idx],j_id[idx]] += ones

    if n=="pseudo" or n=="correlation":# 512x512 0.5*S_inverse image
    S_inv = np.zeros((512,512,4,4),dtype=np.complex64)
    # Build S-Inverse
    del(idx)
        i_id = [0,0,1,1,2,2,3,3]
        j_id = [0,1,2,3,2,3,0,1]
        
        for idx in range(len(i_id)):
            if i_id[idx] == 2 and j_id[idx] == 2:
                S_inv[:,:,i_id[idx],j_id[idx]] -= comps
            
            if i_id[idx] == 2 and j_id[idx] == 3:
                S_inv[:,:,i_id[idx],j_id[idx]] += comps
            
            if i_id[idx] == 3 and j_id[idx] == 1:
                S_inv[:,:,i_id[idx],j_id[idx]] -= ones
            
            if i_id[idx] == 0 and j_id[idx] == 0:
                S_inv[:,:,i_id[idx],j_id[idx]] += ones
            
            if i_id[idx] == 0 and j_id[idx] == 1:
                S_inv[:,:,i_id[idx],j_id[idx]] += ones
            
            if i_id[idx] == 1 and j_id[idx] == 2:
                S_inv[:,:,i_id[idx],j_id[idx]] += ones
            
            if i_id[idx] == 1 and j_id[idx] == 3:
                S_inv[:,:,i_id[idx],j_id[idx]] += ones
            
            if i_id[idx] == 3 and j_id[idx] == 0:
                S_inv[:,:,i_id[idx],j_id[idx]] += ones
    
    return {"S_matrix":S_, "inverse_S_matrix":S_inv}

def MuellerMatrixElement_Pseudo(J,i,j,trans):
    
    Pi = StokesMatrix(i)
    Pj = StokesMatrix(j)
    
    SMatrix= S_Matrix(trans)
    S_inverse = SMatrix["inverse_S_matrix"]
    S = SMatrix["S_matrix"]
    
    # Mueller Matrix elements
    M_ij = np.einsum('...ab,...bc,...cd,...ad',Pi,J,Pj,J.conj()) / 2.
    
    # Mueller Matrix * S_inverse
    MS_inv = [[sum(a*b for a,b in zip(M_row,Sinv_col)) for Sinv_col in zip(*S_inverse)] for M_row in M_ij]
    
    # S * Mueller Matrix * S_inverse
    SMS_inv = [[sum(a*b for a,b in zip(S_row,MSinv_col)) for MSinv_col in zip(*MS_inv)] for S_row in S]
    
    Map_ij = np.real(SMS_inv)
    
    return M_ij

nu0 = 110 #MHz
print "nu0 = ", nu0
nu_nodes = range(nu0-5,nu0+6)

input_files = [txtname(n) for n in nu_nodes]

iJMF = InstrumentalJonesMatrixField(input_files, nu_nodes)

lat = 120.7215
z0_cza = np.radians(lat)
z0 = scbt.r_hat(z0_cza, 0.)

RotAxis = np.array([0,-1,0])
RotAngle = z0_cza

R_z0 = scbt.rotation_matrix(RotAxis, RotAngle)
nu_axis = np.linspace(nu0-1.,nu0+1., 3, endpoint=True)

# BUILD PROPER GRID SIZE
# Determine the proper theta (rad) and phi (rad) values for a Sky Coordinate
# value at a specific HealPix pixel value.
npix_sq = 512
fitsobject = "/Users/tashaleebillings/Desktop/research/data/gc.2457548.uvcRP.forceQUV2zero_Interactive.image.fits"

# Load the FITS hdulits
data, header = fits.getdata(fitsobject, header=True)

# Parse the WCS keywords in the primary HDU
w = wcs.WCS(header)

# Convert Equitorial Coordinates to Spherical Coordinates grid
xpix, ypix = np.meshgrid(np.arange(npix_sq),np.arange(npix_sq), indexing='xy')#np.meshgrid(np.arange(1,npix+1),np.arange(1,npix+1), indexing='xy')
ra, dec, dummy_freq, dummy_stokes = w.all_pix2world(xpix, ypix,1,1,1) #Right ascension and declination as seen on the inside of the celestial sphere
c_icrs = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs') # Coordinates Object

# Convert ra/dec to phi/theta
# don't use ravel because altering the values could change the original data.
theta_grid = np.pi/2. - c_icrs.dec.radian # polar angle limits [0,np.pi]
theta_flat = np.reshape(theta_grid,-1)

#phi_grid = np.radians(header['CRVAL1']) - c_icrs.ra.radian # azimuthal angle limits [0,2*np.pi]
phi_grid = c_icrs.ra.radian.mean() - c_icrs.ra.radian
phi_flat = np.reshape(phi_grid,-1)

# Calculate rotation around -y (?) to get the beam to point at declination = latitude
z0_cza = np.radians(120.7215)
RotAxis = np.array([0,-1,0])
RotAngle = z0_cza
R_z0 = scbt.rotation_matrix(RotAxis, RotAngle)

# Calculcate the theta, phi corresponding to the original coordinate system

theta_hor, phi_hor = scbt.spherical_coordinates_map(R_z0, theta_flat, phi_flat)
phi_hor = 2.*np.pi - phi_hor
ijones_sq = np.reshape(iJMF(nu_axis, theta_hor, phi_hor, R_z0.T),(len(nu_axis),npix_sq, npix_sq, 2, 2))

# Generate Simulation of the Mueller Matrix
i_index = 4
j_index = 4
nchan = len(nu_axis)
npix = 512

MuellerMatrixij_Pseudo = np.zeros((i_index,j_index,nchan,npix,npix),dtype=np.float64) # [i,j, freq,ra pixel,dec pixel]

#ii_index = [0,0,0,0]#,1,1,1,1,2,2,2,2,3,3,3,3]
#jj_index = [0,1,2,3]#,0,1,2,3,0,1,2,3,0,1,2,3]

#for f in range(ijones_sq.shape[0]):
#    for i in range(i_index):
#        for j in range(j_index):
#            MuellerMatrixij_Pseudo[i,j,f,:,:] = MuellerMatrixElement_Pseudo(ijones_sq[f],i,j,trans="correlation")
#            print i,j

#PrimaryHDU
hdu = fits.PrimaryHDU(data=MuellerMatrixij_Pseudo,header=header)
for k in header.keys():
    if k == '':
        continue
    if k == 'HISTORY':
        continue
    hdu.header.append((k,header[k]))

# write new GSM model out to a FITS file
mb_filename='/Users/tashaleebillings/Desktop/research/data/Pseudo_MuellerMatrixBeam_{}MHz.fits'.format("109")
hdu.writeto(mb_filename,overwrite=True)
mb = fits.open(mb_filename)
mb[0].header['CTYPE4']='Mueller Components' # Change the name
mb.writeto(mb_filename,overwrite=True)
