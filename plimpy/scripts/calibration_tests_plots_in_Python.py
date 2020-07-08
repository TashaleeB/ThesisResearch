#!/bin/sh

#  calibration_tests.py
#
#
#  Created by Tashalee Billings on 6/7/18.
#

"""
    After running python script "calibration_test_in_CASA.py", you can run this script to make plots of the calibration solutions. The antenna number must match. You must double check.
   
"""

# Location of the two types of simulated data
# /data4/paper/SimulatedData/HERA19_GSM2008_NicCST2016_full_day/D_term_corrupted/
# /data4/paper/SimulatedData/HERA19_GSM2008_NicCST2016_full_day/true_visibilities/

import numpy as np, matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
import os, glob, sys, aplpy
import time
import h5py

from astropy import units as u
from astropy.io import fits
from matplotlib import gridspec

Knpzlist = glob.glob("*.K.cal.npz") #(2,1,113)
Bnpzlist = glob.glob("*.B.cal.npz") #(2,1024,113)
Gnpzlist = glob.glob("*.G.cal.npz") #solint=int then dim =(2, 1, 6893) or solint=inf then dim =(2,1,113)

NOT_REAL_ANTS="0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 90, 91, 92, 93, 94, 95, 98, 99, 100, 101, 102, 103, 106, 107, 108, 109, 110, 111"#HERA-19

ex_ants = [int(a) for a in NOT_REAL_ANTS.split(', ')] # Turns str to list
keep_ants = [a for a in range(np.load(Knpzlist[0])['gains'].shape[2]) if a not in ex_ants] # Equivalent to having an if no statement in for loop. You are looping over a list.
badant=[22,43,80,81] # Flagged antenna for HERA19
good_ants = [a for a in keep_ants if a not in badant]
ant = [10, 11, 21, 23, 32, 44, 54, 65, 66, 73, 81, 82, 89, 90, 97, 98, 105, 106, 113]
antindex = [9, 10, 20, 22, 31, 43, 53, 64, 65, 72, 80, 81, 88, 89, 96, 97, 104, 105, 112] #position of ant list?

#***********************************************
#-----------------------------------------------
# Create and save .NPZ plots
#-----------------------------------------------
#***********************************************

for kc in Knpzlist:
    k=np.load(kc)["gains"]

    plt.figure(figsize=(14,8))

    plt.plot(k[0,0,:],'.')
    plt.title("{} K Values: East Pol".format(kc))
    plt.xlabel('Antenna Number')
    plt.ylabel('Amp [ns]')
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    plt.grid(which='minor', linestyle='-', linewidth='0.5', color='black')
    
    plt.savefig("KGB_DELAY_amp_east_{}.png".format(kc))
    plt.close()

    plt.figure(figsize=(14,8))
    plt.plot(k[1,0,:],'.')
    plt.title("{} K Values: North Pol".format(kc))
    plt.xlabel('Antenna Number')
    plt.ylabel('Amp [ns]')
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    plt.grid(which='minor', linestyle='-', linewidth='0.5', color='black')
    
    plt.savefig("KGB_DELAY_amp_north_{}.png".format(kc))
    plt.close()

print("Done with Delay Plots.")

"""
for gc in Gnpzlist:
    g=np.load(gc)["gains"]
    #--------------------------------------------------
    # Create Amplitude East and North Plots vs Channel
    #--------------------------------------------------
    plt.figure(figsize=(14,8))
    for i in range(len(ant)):
        if i < 10:
            plt.plot(np.abs(g[0,:,ant[i]-1]),'--',label=str(ant[i]))
        else:
            plt.plot(np.abs(g[0,:,ant[i]-1]),'.',label=str(ant[i]))

    plt.title("{} Amplitude:East Gain Solutions".format(gc))
    plt.xlabel('Antenna Number')
    plt.ylabel('Amplitude')
    plt.legend()
    #plt.show()

    plt.savefig("amp_east_{}.png".format(gc))
    plt.close()

    plt.figure(figsize=(14,8))
    for i in range(len(ant)):
        if i < 10:
            plt.plot(np.abs(b[1,:,ant[i]-1]),'--',label=str(ant[i]))
        else:
            plt.plot(np.abs(b[1,:,ant[i]-1]),'.',label=str(ant[i]))
    plt.title("{} Amplitude:North Gain Solutions".format(gc))
    plt.xlabel('Antenna Number')
    plt.ylabel('Amplitude')
    plt.legend()
    #plt.show()

    plt.savefig("amp_north_{}.png".format(gc))
    plt.close()

    #--------------------------------------------------
    # Create Phase East and North Plots vs Channel
    #--------------------------------------------------
    plt.figure(figsize=(14,8))
    for i in range(len(ant)):
        if i < 10:
            plt.plot(np.angle(g[0,:,ant[i]-1],deg=True),'--',label=str(ant[i]))
        else:
            plt.plot(np.angle(g[0,:,ant[i]-1],deg=True),'.',label=str(ant[i]))

    plt.title("{} Phase:East Gain Solutions".format(gc))
    plt.xlabel('Antenna Number')
    plt.ylabel('Phase [deg]')
    plt.legend()
    #plt.show()

    plt.savefig("phase_east_{}.png".format(gc))
    plt.close()

    plt.figure(figsize=(14,8))
    for i in range(len(ant)):
        if i < 10:
            plt.plot(np.angle(g[1,:,ant[i]-1],deg=True),'--',label=str(ant[i]))
        else:
            plt.plot(np.angle(g[1,:,ant[i]-1],deg=True),'.',label=str(ant[i]))
    plt.title("{} Phase:North Gain Solutions".format(gc))
    plt.xlabel('Antenna Number')
    plt.ylabel('Phase [deg]')
    plt.legend()
    #plt.show()

    plt.savefig("phase_north_{}.png".format(gc))
    plt.close()

print("Done with Gain Type G Plots.")

"""

for bc in Bnpzlist:
    bb =np.load(bc)["gains"]
    b=bb.copy()
    #--------------------------------------------------
    # Create Amplitude East and North Plots vs Channel
    #--------------------------------------------------
    plt.figure(figsize=(14,8))
    for i in range(len(ant)):
        mean= 1#np.mean(np.abs(b[0,:,ant[i]-1]),axis=0) #1
        if i < 10:
            plt.plot(np.abs(b[0,:,antindex[i]])/mean,'--',label=str(ant[i]))
        else:
            plt.plot(np.abs(b[0,:,antindex[i]])/mean,'.',label=str(ant[i]))

    plt.title("{} Amplitude:East Gain Solutions".format(bc))
    plt.xlabel('Channel')
    plt.ylabel('Amplitude')
    plt.legend()
    #plt.show()

    plt.savefig("amp_east_{}.png".format(bc))
    plt.close()

    plt.figure(figsize=(14,8))
    for i in range(len(ant)):
        mean= 1#np.mean(np.abs(b[1,:,ant[i]-1]),axis=0) #1
        if i < 10:
            plt.plot(np.abs(b[1,:,ant[i]-1])/mean,'--',label=str(ant[i]))
        else:
            plt.plot(np.abs(b[1,:,ant[i]-1])/mean,'.',label=str(ant[i]))
    plt.title("{} Amplitude:North Gain Solutions".format(bc))
    plt.xlabel('Channel')
    plt.ylabel('Amplitude')
    plt.legend()
    #plt.show()

    plt.savefig("amp_north_{}.png".format(bc))
    plt.close()

    #--------------------------------------------------
    # Create Phase East and North Plots vs Channel
    #--------------------------------------------------
    plt.figure(figsize=(14,8))
    for i in range(len(ant)):
        if i < 10:
            plt.plot(np.angle(b[0,:,ant[i]-1],deg=True),'--',label=str(ant[i]))
        else:
            plt.plot(np.angle(b[0,:,ant[i]-1],deg=True),'.',label=str(ant[i]))

    plt.title("{} Phase:East Gain Solutions".format(bc))
    plt.xlabel('Channel')
    plt.ylabel('Phase [deg]')
    plt.legend()
    #plt.show()

    plt.savefig("phase_east_{}.png".format(bc))
    plt.close()

    plt.figure(figsize=(14,8))
    for i in range(len(ant)):
        if i < 10:
            plt.plot(np.angle(b[1,:,ant[i]-1],deg=True),'--',label=str(ant[i]))
        else:
            plt.plot(np.angle(b[1,:,ant[i]-1],deg=True),'.',label=str(ant[i]))
    plt.title("{} Phase:North Gain Solutions".format(bc))
    plt.xlabel('Channel')
    plt.ylabel('Phase [deg]')
    plt.legend()
    #plt.show()

    plt.savefig("phase_north_{}.png".format(bc))
    plt.close()

print("Done with Bandpass Plots.")


print("Creating STOKES Visibility Plots.")
#***********************************************
#-----------------------------------------------
# Create and save .fits plots
#-----------------------------------------------
#***********************************************
files_from_fits = glob.glob("*.fits")

for fitsfile in files_from_fits:

    f = plt.figure(figsize=(10,8))
    for pol in np.arange(4):
        fig = aplpy.FITSFigure(fitsfile,dimensions=[0,1],slices=[0,pol],figure=f,subplot=(2,2,pol+1))
        if pol == 0:
            vmax=1.7
            vmin=-0.2
            cmap="viridis"
        if pol == 1:
            vmax=0.3
            vmin=-.2
            cmap="PRGn" #"RdYlGn"
        if pol == 2:
            vmax=0.1
            vmin=-0.03
            cmap="PRGn"
        if pol == 3:
            vmax=0.003
            vmin=-0.007
            cmap="PRGn"
        
        fig.show_colorscale(cmap=cmap)#,vmax=vmax,vmin=vmin)#,stretch='arcsczdxcinh')
        fig.add_grid()
        fig.grid.set_color('black')
        fig.grid.set_xspacing(15)
        fig.grid.set_yspacing(15)
        fig.grid.show()
        fig.axis_labels.set_font(size='small')
        fig.tick_labels.set_font(size='small')
        fig.tick_labels.set_xformat('hh')
        fig.tick_labels.set_yformat('dd')
        fig.add_colorbar()
        fig.colorbar.set_font(size='small')

    plt.suptitle('{} STOKE Visibilities'.format(fitsfile))
    plt.tight_layout()
    fig.savefig('{}.STOKES.png'.format(fitsfile))
    plt.close()


#***********************************************
#-----------------------------------------------
# Compare CASA Solutions to Zac
#-----------------------------------------------
#***********************************************
path2hdf5 = "/lustre/aoc/projects/hera/tbilling/polim/calibration/sims/gains_only/trial3/gains_only_Solutions/"
hdf5file = "gains_and_D_terms.h5"

freqs = np.linspace(100,200,num=1024) # Units of MHz

# http://docs.h5py.org/en/latest/quick.html

f = h5py.File(path2hdf5+hdf5file,'r')
#f = h5py.File(hdf5file,'r')
#print(f.keys())
gains_e = f['gains_e'].value
gains_n = f['gains_n'].value

#--------------------------------------------------
# Geometric Delay
#--------------------------------------------------

for kc in Knpzlist:
    k=np.load(kc)["gains"]
    karray = k[:,0,antnum] # Dim (2,19)
    
    nrow,ncol = 3,5
    numplots = nrow*ncol
    iarr = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2]
    jarr = [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]
 
 #East Phase Comparison
    f,axarr = plt.subplots(nrow,ncol,sharex=True,sharey=True,figsize=(15,8))
    f.suptitle("East Geometric Delay: {}".format(kc), fontsize=14)#Title centered above all subplots
    gs = gridspec.GridSpec(nrow,ncol)
    
    for i,a in enumerate(good_ants):
        index = np.where(np.array(antnum)==a)[0][0]
        tauX = karray[0,index]*1e-9 #units of sec
        phi_Xdelay= 2*np.pi*tauX*(freqs*1e6) # Units of Rad

        ax=plt.subplot(gs[iarr[i],jarr[i]])
        #ax.set_ylim(-np.pi,np.pi)
        ax.set_xlim(100,200)
        ax.annotate(str(a), xy=(187,0.05), size=9, color='deeppink')
        #ax.fill_between(freqs[201:300],-0.2,3.2, facecolor='k',alpha=0.2)
        #ax.fill_between(freqs[581:680],-0.2,3.2,facecolor='k',alpha=0.2)
    
        if i > 9:
            ax.set_xlabel('Frequency [MHz]',size=12)
            ax.grid(color='k', linestyle='--', linewidth=1)
        if i == 0:
            ax.set_ylabel('Phase [Radian]',size=12)
            ax.plot(freqs,np.angle(gains_e[index,:], deg=False),'k-',lw=2,label="Zac Solution")
            ax.plot(freqs,phi_Xdelay,'r.')
        if i == 5:
            ax.set_ylabel('Phase [Radian]',size=12)
            ax.plot(freqs,np.angle(gains_e[index,:], deg=False),'k-',lw=2)
            ax.plot(freqs,phi_Xdelay,'r.')
        if i == 10:
            ax.set_ylabel('Phase [Radian]',size=12)
            ax.plot(freqs,np.angle(gains_e[index,:], deg=False),'k-',lw=2)
            ax.plot(freqs,phi_Xdelay,'r.')
        
        else:
            ax.grid(color='k', linestyle='--', linewidth=1)
            ax.plot(freqs,np.angle(gains_e[index,:], deg=False),'k-',lw=2)
            ax.plot(freqs,phi_Xdelay,'r.')

        print("Tau East: "+str(tauX) +" Hz")

    ax.legend(bbox_to_anchor=(1.6, 1), loc='lower right', borderaxespad=0.)

    f.savefig("{}_Compare_Tau_to_Zac_Solution_per_antenna_east.png".format(kc))
    print("{}_Compare_Tau_to_Zac_Solution_per_antenna_east.png".format(kc))
    plt.close()

#North Phase Comparison
    f,axarr = plt.subplots(nrow,ncol,sharex=True,sharey=True,figsize=(15,8))
    f.suptitle("North Geometric Delay: {}".format(kc), fontsize=14)#Title centered above all subplots
    gs = gridspec.GridSpec(nrow,ncol)
    
    for i,a in enumerate(good_ants):
        index = np.where(np.array(antnum)==a)[0][0]
        tauY = karray[1,index]*1e-9 #units of sec
        phi_Ydelay= 2*np.pi*tauY*(freqs*1e6) # Units of Radian
        
        ax=plt.subplot(gs[iarr[i],jarr[i]])
        #ax.set_ylim(-np.pi,np.pi)
        ax.set_xlim(100,200)
        ax.annotate(str(a), xy=(187,0.05), size=9, color='deeppink')
        #ax.fill_between(freqs[201:300],-0.2,3.2, facecolor='k',alpha=0.2)
        #ax.fill_between(freqs[581:680],-0.2,3.2,facecolor='k',alpha=0.2)
        
        if i > 9:
            ax.set_xlabel('Frequency [MHz]',size=12)
            ax.grid(color='k', linestyle='--', linewidth=1)
        if i == 0:
            ax.set_ylabel('Phase [Radian]',size=12)
            ax.plot(freqs,np.angle(gains_n[index,:], deg=False),'k-',lw=2)
            ax.plot(freqs,phi_Ydelay,'r.')
        if i == 5:
            ax.set_ylabel('Phase [Radian]',size=12)
            ax.plot(freqs,np.angle(gains_n[index,:], deg=False),'k-',lw=2)
            ax.plot(freqs,phi_Ydelay,'r.')
        if i == 10:
            ax.set_ylabel('Phase [Radian]',size=12)
            ax.plot(freqs,np.angle(gains_n[index,:], deg=False),'k-',lw=2)
            ax.plot(freqs,phi_Ydelay,'r.')
        
        else:
            ax.grid(color='k', linestyle='--', linewidth=1)
            ax.plot(freqs,np.angle(gains_n[index,:], deg=False),'k-',lw=2)
            ax.plot(freqs,phi_Ydelay,'r.')

        print("Tau North: "+str(tauY) +" Hz")

    ax.legend(bbox_to_anchor=(1.6, 1), loc='lower right', borderaxespad=0.)

    f.savefig("{}_Compare_Tau_to_Zac_Solution_per_antenna_north.png".format(kc))
    print("{}_Compare_Tau_to_Zac_Solution_per_antenna_north.png".format(kc))
    plt.close()

#--------------------------------------------------
# Bandpass Solutions
#--------------------------------------------------

for bc in Bnpzlist:
    b=np.load(bc)["gains"]
    barray = b[:,:,antnum] # Dim (2,1024,19)
    
    nrow,ncol = 3,5
    numplots = nrow*ncol
    iarr = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2]
    jarr = [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]
    
    #East Phase Comparison
    f,axarr = plt.subplots(nrow,ncol,sharex=True,sharey=True,figsize=(15,8))
    f.suptitle("East-Pol Bandpass Solutions: {}".format(bc), fontsize=14)#Title centered above all subplots
    gs = gridspec.GridSpec(nrow,ncol)
    
    for i,a in enumerate(good_ants):
        index = np.where(np.array(antnum)==a)[0][0]
        ax=plt.subplot(gs[iarr[i],jarr[i]])
        #ax.set_ylim(-np.pi,np.pi)
        ax.set_xlim(100,200)
        ax.annotate(str(a), xy=(187,0.05), size=9, color='deeppink')
        #ax.fill_between(freqs[201:300],-0.2,3.2, facecolor='k',alpha=0.2)
        #ax.fill_between(freqs[581:680],-0.2,3.2,facecolor='k',alpha=0.2)
        
        if i > 9:
            ax.set_xlabel('Frequency [MHz]',size=12)
            ax.grid(color='k', linestyle='--', linewidth=1)
        if i == 0:
            ax.set_ylabel('Amp [Arb.]',size=12)
            ax.plot(freqs,np.abs(gains_e[index,:]),'k-',lw=2,label="Zac Solution")
            ax.plot(freqs,np.abs(b[0,:,a]),'r.')
        if i == 5:
            ax.set_ylabel('Amp [Arb.]',size=12)
            ax.plot(freqs,np.abs(gains_e[index,:]),'k-',lw=2)
            ax.plot(freqs,np.abs(b[0,:,a]),'r.')
        if i == 10:
            ax.set_ylabel('Amp [Arb.]',size=12)
            ax.plot(freqs,np.abs(gains_e[index,:]),'k-',lw=2)
            ax.plot(freqs,np.abs(b[0,:,a]),'r.')
        
        else:
            ax.grid(color='k', linestyle='--', linewidth=1)
            ax.plot(freqs,np.abs(gains_e[index,:]),'k-',lw=2)
            ax.plot(freqs,np.abs(b[0,:,a]),'r.')

    ax.legend(bbox_to_anchor=(1.6, 1), loc='lower right', borderaxespad=0.)

    f.savefig("{}_Compare_Bandpass_to_Zac_Solution_per_antenna_east.png".format(bc))
    print("{}_Compare_Bandpass_to_Zac_Solution_per_antenna_east.png".format(bc))
    plt.close()

    #North Phase Comparison
    f,axarr = plt.subplots(nrow,ncol,sharex=True,sharey=True,figsize=(15,8))
    f.suptitle("North-Pol Bandpass Solutions: {}".format(bc), fontsize=14)#Title centered above all subplots
    gs = gridspec.GridSpec(nrow,ncol)

    for i,a in enumerate(good_ants):
        index = np.where(np.array(antnum)==a)[0][0]
        ax=plt.subplot(gs[iarr[i],jarr[i]])
        #ax.set_ylim(-np.pi,np.pi)
        ax.set_xlim(100,200)
        ax.annotate(str(a), xy=(187,0.05), size=9, color='deeppink')
        #ax.fill_between(freqs[201:300],-0.2,3.2, facecolor='k',alpha=0.2)
        #ax.fill_between(freqs[581:680],-0.2,3.2,facecolor='k',alpha=0.2)

        if i > 9:
            ax.set_xlabel('Frequency [MHz]',size=12)
            ax.grid(color='k', linestyle='--', linewidth=1)
        if i == 0:
            ax.set_ylabel('Amp [Arb.]',size=12)
            ax.plot(freqs,np.abs(gains_n[0,:]),'k-',lw=2)
            ax.plot(freqs,np.abs(b[1,:,a]),'r.')
        if i == 5:
            ax.set_ylabel('Amp [Arb.]',size=12)
            ax.plot(freqs,np.abs(gains_n[0,:]),'k-',lw=2)
            ax.plot(freqs,np.abs(b[1,:,a]),'r.')
        if i == 10:
            ax.set_ylabel('Amp [Arb.]',size=12)
            ax.plot(freqs,np.abs(gains_n[0,:]),'k-',lw=2)
            ax.plot(freqs,np.abs(b[1,:,a]),'r.')

        else:
            ax.grid(color='k', linestyle='--', linewidth=1)
            ax.plot(freqs,np.abs(gains_n[0,:]),'k-',lw=2)
            ax.plot(freqs,np.abs(b[1,:,a]),'r.')

    ax.legend(bbox_to_anchor=(1.6, 1), loc='lower right', borderaxespad=0.)

    f.savefig("{}_Compare_Bandpass_to_Zac_Solution_per_antenna_north.png".format(bc))
    print("{}_Compare_Bandpass_to_Zac_Solution_per_antenna_north.png".format(bc))
    plt.close()









