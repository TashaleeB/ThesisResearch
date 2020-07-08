import numpy as np
import time
import shutil
import os, glob, sys


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

def delay(namems):
    calitable = os.path.basename(namems) + ".K.cal"
    gaincal(namems,caltable=calitable,gaintype = 'K' , solint='inf',refant='10')
    print("Created "+glob.glob(calitable)[0])
    applycal(namems,gaintable=[calitable])

    return

def bandpasscal(namems):
    calitable = os.path.basename(namems) +".B.cal"
    bandpass(namems,caltable=calitable,solint='inf',combine='scan',refant='10')
    print("Created "+glob.glob(calitable)[0])
    applycal(namems,gaintable=[calitable])
    
    return

def clean1(namems):
    clean(namems,namems[:-5]+'_no_cal_niter0',niter=0, weighting = 'briggs',robust =0, imsize =[512 ,512] ,pbcor=False, cell=['500 arcsec'] ,mode='mfs',nterms=1, spw='0:150~900', stokes='IQUV',interactive=False, npercycle=5, threshold='0.1mJy/beam')

    return

calfits_model='old_GSM_Model_of_GC_Jy_1024StokesIQUV_WithHeaderCopiedToIt.fits'
cal_model = calfits_model.strip('fits')+'image'

# Convert FITS image to CASA image format.
importuvfits(calfits_model, cal_model)

# Convert UVFITS file to MS
uvfits = glob.glob('gc.2457548.uvcRP.uvfits')
for uvfit in uvfits:
msfile = 'GSM_'+uvfit.strip('uvfits') + 'MS'
importuvfits(vis=msfile,fitsfile=uvfit)

# Flag bad data
flag(msfile)
# Insert image cube into MS
imagemodel(msfile,cal_model)

# Start calibrtion Process
delay(msfile) # Delay Calibration
bandpasscal(msfile) # Bandpass Calibration

# Dirty Image of Calibrated Data
clean(msfile,msfile[:-5]+'_KBcal_NoDeconv',niter=0, weighting = 'briggs',robust =0,
      imsize =[512 ,512] ,pbcor=False, cell=['500 arcsec'] ,
      mode='mfs',nterms=1, spw='0:150~900', stokes='IQUV',
      interactive=False, npercycle=5, threshold='0.1mJy/beam')
exportfits(msfile[:-5]+'_KBcal_NoDeconv.image',msfile[:-5]+'_KBcal.image.fits')
