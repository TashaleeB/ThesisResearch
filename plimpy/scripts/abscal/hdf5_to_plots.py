#!/bin/sh

#  hdf5_to_plots.py
#  
#
#  Created by Tashalee Billings on 10/15/18.
#  
from __future__ import print_function, division, absolute_import
from pyuvdata import UVData, UVCal
import numpy as np
import os,glob
import h5py

f = h5py.File('../../gains_and_D_terms.h5', 'r')
#list(f.keys())
#[u'D_terms_flat', u'D_terms_wiggly', u'gains_e', u'gains_n']
ant = [10, 11, 21, 23, 32, 44, 54, 65, 66, 73, 81, 82, 89, 90, 97, 98, 105, 106, 113]

solulist = list(f.keys())

for sl in solulist:
    hdf5_data =f[sl]
    #--------------------------------------------------
    # Create Amplitude East and North Plots vs Channel
    #--------------------------------------------------
    plt.figure(figsize=(14,8))
    for i in range(hdf5_data.shape[0]):
        plt.plot(np.abs(hdf5_data[i,:]),'--',label=str(ant[i]))

    plt.title("{} Amplitude: Gain Solutions".format(sl))
    plt.xlabel('Channel')
    plt.ylabel('Amplitude')
    plt.legend()
    #plt.show()

    plt.savefig("amp_{}.png".format(sl))
    plt.close()

#--------------------------------------------------
# Create Phase East and North Plots vs Channel
#--------------------------------------------------
    plt.figure(figsize=(14,8))
    for i in range(hdf5_data.shape[0]):
        plt.plot(np.angle(hdf5_data[i,:],deg=True),'--',label=str(ant[i]))

    plt.title("{} Phase: Gain Solutions".format(sl))
    plt.xlabel('Channel')
    plt.ylabel('Phase [deg]')
    plt.legend()
    #plt.show()

    plt.savefig("phase_{}.png".format(sl))
    plt.close()

print("Done with Bandpass Plots.")
