#!/bin/sh

#  cst2jones_test.py
#  
#
#
#

import numpy as np, healpy as hp, matplotlib.pyplot as plt
import os,sys
import colorcet as cc
ccc = cc.m_cyclic_grey_15_85_c0_s25
import matplotlib
import cmocean

from mpl_toolkits.axes_grid1 import AxesGrid
from astropy.io import fits
from astropy import wcs

sys.path.insert(0,'/users/tbilling/')
from cst2ijones.jones_matrix_field  import InstrumentalJonesMatrixField
from cst2ijones.plotting import PlotMueller
from cst2ijones import spherical_coordinates_basis_transformation as scbt

def StokesMatrix(n):
    if n not in [0,1,2,3]: raise Exception('Input must be an integer in [0,1,2,3]')
    
    if n == 0:
        p = np.array([[1.,0],[0.,1.]])
    elif n == 1:
        p = np.array([[1.,0],[0,-1.]])
    elif n == 2:
        p = np.array([[0,1.],[1.,0]])
    elif n == 3:
        p = np.array([[0., -1j],[1j,0]])
    
    return p

def MuellerMatrixElement(J,i,j):
    
    Pi = StokesMatrix(i)
    Pj = StokesMatrix(j)
    
    M_ij = np.einsum('...ab,...bc,...cd,...ad',Pi,J,Pj,J.conj()) / 2.
    
    M_ij = np.real(M_ij)
    
    return M_ij

def PlotMueller(jones):
    npix = jones.shape[0]
    nside = hp.npix2nside(npix)
    xsize = 1600
    reso = 120*180*np.sqrt(2.)/np.pi /(xsize-1)
    LambProj = hp.projector.AzimuthalProj(xsize=xsize,reso=reso, lamb=True, half_sky=True, rot=[0,30.72])
    p2v = lambda x,y,z: hp.vec2pix(nside,x,y,-z)
    
    logthresh = 4
    linscale = 2
    fig = plt.figure(figsize=(12,12))
    grid = AxesGrid(fig,(1,1,1),
                    nrows_ncols=(4,4),
                    axes_pad=(1.0,0.5),
                    label_mode='all',
                    share_all=False,
                    cbar_location='right',
                    cbar_mode='each',
                    cbar_size='5%',
                    cbar_pad='1%',
                    )
    for i in range(4):
        for j in range(4):
            M_ij = MuellerMatrixElement(jones, i, j)/2.
            img_d = LambProj.projmap(M_ij, p2v)

            if i == j == 0:
                cmap = 'viridis'
                vmin = 0
                vmax = 1

                tick_locs = list(np.linspace(0,1,7, endpoint=True))
                tick_labels = [r'$ < 10^{-6}$',
                               r'$10^{-5}$',
                               r'$10^{-4}$',
                               r'$10^{-3}$',
                               r'$10^{-2}$',
                               r'$10^{-1}$',
                               r'$10^{0}$']

            elif i != j:
                cmap='RdBu_r'
                vmin=-0.05
                vmax=0.05

                d = np.log10(5) * np.diff(np.linspace(vmax*1e-6,vmax,7))[0]
                q = np.linspace(vmax*1e-6,vmax,7)[0::2] - d
                tick_locs = list(np.r_[-np.flipud(q)[:-1],[0], q[1:]])
                tick_labels = [r'$-10^{-2}$',
                               r'$-10^{-4}$',
                               r'$-10^{-6}$',
                               r'$< 5 \times 10^{-8}$',
                               r'$10^{-6}$',
                               r'$10^{-4}$',
                               r'$10^{-2}$']

            else:
                cmap=cmocean.cm.delta
                vmin=-1.
                vmax=1

                q = np.linspace(vmax*1e-6, vmax,7)[0::2]
                tick_locs = list(np.r_[-np.flipud(q)[:-1],[0], q[1:]])
                tick_labels = [r'$-10^{0}$',
                               r'$-10^{-2}$',
                               r'$-10^{-4}$',
                               r'$< 10^{-6}$',
                               r'$10^{-4}$',
                               r'$10^{-2}$',
                               r'$10^{0}$']
            n = 4 * i + j
            im = grid[n].imshow(img_d, interpolation='none',
            cmap=cmap,
            aspect='equal',
            vmin=vmin,
            vmax=vmax,)

            grid[n].set_xticks([])
            grid[n].set_yticks([])

            cbar = grid.cbar_axes[n].colorbar(im, ticks=tick_locs)
            grid.cbar_axes[n].set_yticklabels(tick_labels)

            im.set_norm(matplotlib.colors.SymLogNorm(10**-logthresh,linscale, vmin=vmin,vmax=vmax))
    plt.tight_layout(w_pad=0.5, h_pad=1.0)
#     plt.savefig('full_mueller_airy_150MHz.png', dpi=80, bbox_inches='tight')
#     plt.savefig('full_mueller_150MHz.pdf', bbox_inches='tight')
#     plt.close(plt.gcf())
#     plt.show()

def txtname(n):
    if n not in range(50,251):
        raise ValueError('no data at that frequency.')
    fname = '/lustre/aoc/projects/hera/tbilling/NF_Simulations/Radiation patterns/E-field pattern - Rigging height 4.9 m/HERA_4.9m_E-pattern_ {0}MHz.txt'.format(str(n))
    return fname

nfreq = 2
npixels = 512 # length of each side of the HealPix image
nu_axis = np.arange(100,106)
full_mueller_matrix = np.zeros((4,4,nfreq,npixels,npixels))

# Load the FITS hdulits
fitsobject = '/lustre/aoc/projects/hera/tbilling/polim/GSM_GC_model/GSM_Model_of_GC_Jy_1024StokesIQUV_WithHeaderCopiedToIt.fits' # Fits file to PBCorr [Stokes, Freq, Ra, Dec]
data, header = fits.getdata(fitsobject, header=True)

input_files = [txtname(n) for n in nu_axis]

# Parse the WCS keywords in the primary HDU
w = wcs.WCS(header)

# Initiate Instrument Jones Field Function for a certain frequency range.
iJMF = InstrumentalJonesMatrixField(input_files, nu_axis)

# Rotation Matrix
z0_cza = np.radians(120.7215)
z0 = scbt.r_hat(z0_cza, 0.)

RotAxis = np.array([0,-1,0])
RotAngle = z0_cza

R_z0 = scbt.rotation_matrix(RotAxis, RotAngle) # Rotation around y axis

# Determine the proper theta (rad) and phi (rad) values for a Sky Coordinate value at a specific HealPix pixel value.

# Build pixel axis
l_axis = np.linspace(-1, 1, num= npixels, endpoint= True) # l = cos(phi)sin(theta)
m_axis = np.linspace(-1, 1, num= npixels, endpoint= True) # m = sin(phi)sin(theta)

# Create Meshgrid from the pixels
l_array, m_array = np.meshgrid(l_axis, m_axis, indexing= 'xy', sparse= False) # Returns a coord Matrix

# Use Trig manipulations to express phi and theta as a function of Sky Coordinates
rr_array = np.power(l_array, 2) + np.power(m_array, 2) # = (Sin(theta))**2

n_array = np.zeros_like(rr_array) # [512,512]
n_array[rr_array <= 1.] = np.sqrt(1 - rr_array[rr_array <= 1.]) # = cos(theta) [-1,1]
theta_array = np.arccos(n_array) # Theta values [0,2*np.pi]

phi_array = np.arctan2(m_array,l_array) # Phi values [0, np.pi]
fix_phi = np.where(phi_array < 0.)[0]
valid_phi_array = np.copy(phi_array)
valid_phi_array[fix_phi] = 2*np.pi + valid_phi_array[fix_phi]
valid_phi_array[valid_phi_array>=np.pi] = np.nan # shouldn't be larger than np.pi

flatten_theta = theta_array.flatten() # don't use ravel because it altering the flattened variable could change the original data.
flatten_valid_phi = valid_phi_array.flatten()

#Interpolate Beam [freq, Npixels, Row_jones_matrix, Colu_jones_matrix]
ijones = iJMF(nu_axis[2:4], flatten_theta, flatten_valid_phi, R_z0.T)
ijones[:,:,:,0] *= -1. # Change sign for all of the rows in the first column.

# We can reshape the iJones Matrix [freq, ra, dec, Row_jones_matrix, Colu_jones_matrix]
flat_ijones = np.zeros((nfreq,npixels**2,2,2), dtype=np.complex)
flat_ijones[:, nn_flat < 1., :, :] = ijones
reshape_ijones = np.reshape(flat_ijones, (nfreq, npixels, npixels, 2, 2))

# Build Mueller Maxtrix components.
for f, freq in enumerate(nu_axis):
    ijones_per_freq = ijones[freq,:,:,:]
    
    for i in range(4):
        for j in range(4):
            M_ij = MuellerMatrixElement(ijones_per_freq, i, j)/2.
            full_mueller_matrix[i,j,freq,:,:] = M_ij

reshape_mueller_matrix = np.reshape(full_mueller_matrix, (16, nfreq, npixels, npixels))
# Build PrimaryHDU
pbcorr_data = np.dot(full_mueller_matrix,data)
hdu = fits.PrimaryHDU(data=pbcorr_data, header=header)

for k in header.keys():
    if k == '':
        continue
    if k == 'HISTORY':
        continue
    hdu.header.append((k,header[k]))

# write new GSM model out to a FITS file
hdu.writeto('MeullerMatrixpbcorr_{}.fits'.format(fitsobject),overwrite=True)


