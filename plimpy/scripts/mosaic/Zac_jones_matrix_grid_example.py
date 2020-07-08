#!/bin/sh

#  Zacs_jones_matrix_image_grid_example.sh
#
#
#  Created by Tashalee Billings on 1/31/19.
#


import numpy as np, matplotlib.pyplot as plt, healpy as hp

from cst2ijones.jones_matrix_field import *
from cst2ijones.spherical_coordinates_basis_transformation import *
from cst2ijones.directivity_field import *

import matplotlib as mpl
mpl.rcParams['image.cmap'] = 'viridis'
mpl.rcParams['image.aspect'] = 'auto'
mpl.rcParams['image.interpolation'] = 'none'


def txtname(n):
    if n not in range(50,251):
        raise ValueError('no data at that frequency.')
    fname = '/home/zmart/usb_drive/HERA 4.9m - E-field/farfield (f={0}) [1].txt'.format(str(n))s

    return fname

file_freqs = range(145,155)

input_files = [txtname(n) for n in file_freqs]

J = InstrumentalJonesMatrixField(input_files, file_freqs, use_multiprocessing=True)

z0_cza = np.radians(120.7215)
z0 = r_hat(z0_cza, 0.)

RotAxis = np.array([0,-1,0])
RotAngle = z0_cza

R_z0 = rotation_matrix(RotAxis, RotAngle)

nu_axis = np.linspace(149.,151.,5,endpoint=True)

J.set_interpolant(nu_axis)

side_pix = 200

l_axis = np.linspace(-0.5,0.5,side_pix, endpoint=True)
m_axis = np.linspace(-0.5,0.5,side_pix, endpoint=True)

ll_grid, mm_grid = np.meshgrid(l_axis, m_axis, indexing='xy')
# ll_grid = np.transpose(ll_grid)
# mm_grid = np.transpose(mm_grid)

rr2_grid = ll_grid**2. + mm_grid**2.

nn_grid = np.zeros_like(ll_grid)
nn_grid[rr2_grid < 1.] = np.sqrt(1. - rr2_grid[rr2_grid < 1.]**2.)

plt.figure()
plt.imshow(nn_grid)
plt.show()

nn_flat = nn_grid.flatten()

ll_grid.shape

ll_use, mm_use = ll_grid.flatten(), mm_grid.flatten()
ll_use = ll_use[nn_flat < 1.]
mm_use = mm_use[nn_flat < 1.]

th_use = np.arccos(nn_flat[nn_flat < 1.])

phi_grid = np.arctan2(mm_grid, ll_grid)

fix_phi = np.where(phi_grid < 0.)[0]
phi_grid[fix_phi] = 2*np.pi + phi_grid[fix_phi]

phi_grid[nn_grid >= 1.] = np.nan

phi_use = (phi_grid.flatten())[nn_flat < 1.]

plt.close(plt.gcf())

plt.imshow(phi_grid, origin='lower')
plt.show()

plt.figure()
plt.imshow(ll_grid, origin='lower')
plt.figure()
plt.imshow(mm_grid, origin='lower')
plt.show()

R_id = np.eye(3)
ijones_init = J(nu_axis, th_use, phi_use, R_id)

ijones_flat = np.zeros((5,side_pix**2, 2, 2), dtype=np.complex)

ijones_flat[:, nn_flat < 1., :, :] = ijones_init

ijones_img = np.reshape(ijones_flat, (5,side_pix, side_pix, 2, 2))

ijones_img.shape

Bee_img = np.abs(ijones_img[...,0,0])**2. + np.abs(ijones_img[...,0,1])**2.

np.amin(Bee_img), np.amax(Bee_img)

plt.imshow(np.log10(Bee_img[2]))
plt.show()

plt.figure()
plt.imshow(np.real(ijones_img[2,:,:,0,0].T), origin='lower')

plt.figure()
plt.imshow(np.real(ijones_img[2,:,:,0,1].T), origin='lower', cmap='RdBu_r')

plt.figure()
plt.imshow(np.real(ijones_img[2,:,:,1,0].T), origin='lower')

plt.figure()
plt.imshow(np.real(ijones_img[2,:,:,1,1].T), origin='lower', cmap='RdBu_r')
plt.show()

