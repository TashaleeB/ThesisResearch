#!/bin/sh

#  flatReal_Dterms_nogains.py
#  
#
#  Created by Tashalee Billings on 10/15/18.
#  
"""
   This script is used with CASA to calibrate simulated data with no gains but real and flat D-Terms.
"""

import os, glob, sys
from recipes.almapolhelpers import *

#Flag the same RFI and bad antennas as Real Data
def flag(namems): #You might have to update this. Flag MS or calibration Tables.
    flagdata(namems, flagbackup=True, mode='manual',antenna="23" ) # 22 for HERA19
    flagdata(namems, flagbackup=True, mode='manual',antenna="44" ) # 43 for HERA19
    flagdata(namems, flagbackup=True, mode='manual',antenna="81" ) # 80 for HERA19
    flagdata(namems, flagbackup=True, mode='manual',antenna="82" ) # 81 for HERA19
    
    flagdata(namems, flagbackup=True, mode='manual', spw="0:0~65" )#channels 0-65 of spw 0
    flagdata(namems, flagbackup=True, mode='manual', spw ="0:930~1023")
    flagdata(namems, flagbackup=True, mode='manual', spw="0:377~387" )
    flagdata(namems, flagbackup=True, mode='manual', spw="0:850~854" )
    flagdata(namems, flagbackup=True, mode='manual', spw = "0:831" )
    flagdata(namems, flagbackup=True, mode='manual', spw = "0:769" )
    flagdata(namems, flagbackup=True, mode='manual', spw = "0:511" )
    flagdata(namems, flagbackup=True, mode='manual', spw = "0:913" )
    
    flagdata(namems, autocorr = True )
    return

def imagemodel(namems, mi): #NAMEofIMAGE.model
    ft(namems,model=mi,usescratch = True) # multiple flat point spectrum model.
    return

def flatspecmodel(namems): # single flat point spectrum model.
    cl.addcomponent(flux=1.0, fluxunit='Jy', shape='point',
                    dir='J2000 17h45m40.0409s -29d0m28.118s')
        
    if os.path.exists("GC.cl"):
        shutil.rmtree("GC.cl")
    cl.rename("GC.cl")
    cl.close()

    ft(namems, complist="GC.cl", usescratch=True)
    return

def Dterms():
    
    
    return

#convert uvfits to MS
uvfits = glob.glob('*.uvfits')
for uvfit in uvfits:
    msfile = uvfit.strip('uvfits') + 'MS'
    importuvfits(vis=msfile,fitsfile=uvfit)
    print(glob.glob("*.MS")[0]+" created.")

#If the XY-phase bandpass is uniformly zero, then the source linear polarization function will occur entirely in the real part of the cross-hand visibilities.

# Determine the Cross-Hand Delay
gaincal(vis='polcal_20080224.cband.all.ms',
        caltable='polcal.xdelcal',
        field='',
        solint='inf',
        combine='scan',
        refant='10',
        smodel=[1.0,0.0,0.0,0.0],
        gaintype='KCROSS',
        gaintable=[])
applycal(msname,gaintable=['polcal.xdelcal'])

# Solve for the average of all baselines together and first solves for a channelized XY-phase (the slope of the source polarization function in the complex plane in each channel), then corrects the slope and solves for a channel-averaged source polarization.
gaincal(vis='polcal_linfeed.ms',
        caltable='polcal.xy0amb',  # possibly with 180deg ambiguity
        field='',                 # the calibrator
        solint='inf',
        combine='scan',
        preavg=200.0,              # minimal parang change
        smodel=[1.0,0.0,0.0,0.0],          # non-zero U assumed
        gaintype='XYf+QU',
        gaintable=[])  # all prior calibration
applycal(msname,gaintable=['polcal.xy0amb'])

# Determine D-Terms Only
polcal(vis= 'polcal_linfeed.ms',
       caltable='polcal.D.cal',
       field='',
       solint='inf',
       combine='scan',
       preavg=200,
       poltype= 'Df',  # 'Dflls' - freq-dep LLS solver or 'Df' - Solve for instrumental polarization (leakage D-terms)
       refant='',            # no reference antenna
       smodel=[1.0,0.0,0.0,0.0],
       gaintable=[])
applycal(msname,gaintable=['polcal.D.cal'])
