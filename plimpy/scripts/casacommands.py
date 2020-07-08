#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
    Created on Thu Nov  2 17:18:23 2017
    
    @author: tashalee
    """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob,os

from astropy.io import fits #for python only
from fitsconverter import * #for python only

#printf "\e[?2004l"

#Convert from .uvfits to .MS
#*******************************************************************
import glob
uvfits = glob.glob('*.uvfits')
for uvfit in uvfits:
    msfile=uvfit.strip('uvfits') + 'MS'
    importuvfits(vis=msfile,fitsfile=uvfit)
    
# try out uvfit.split('.')[:-1]
#*******************************************************************


def reorder(msname):
    import casac
    ms=casac.casac.table()
    ms.open(msname,nomodify=False)
    a1 , a2 , data = [ms.getcol(x) for x in [ "ANTENNA1" , "ANTENNA2" , "DATA" ]]
    m = a1 > a2 #Creates array of True/False values. If m is all false then you get an empty array.
    data [: ,: ,m]= data [: ,: , m ]. conj ()
    x = a2 [ m ]
    a2 [ m ]= a1 [ m ]
    a1 [ m ]= x
    ms.putcol("ANTENNA1",a1)
    ms.putcol("ANTENNA2",a2)
    ms.putcol("DATA",data)
    ms.flush ()
    ms.close ()

def flag(msname): #You might have to update this. Flag MS or calibration Tables.
    flagdata(msname, flagbackup=True, mode='manual',antenna="50" ) #for Fornax
#    flagdata(msname, flagbackup=True, mode='manual',antenna="22" ) #for HERA19
#    flagdata(msname, flagbackup=True, mode='manual',antenna="43" ) #for HERA19
#    flagdata(msname, flagbackup=True, mode='manual',antenna="80" ) #for HERA19
#    flagdata(msname, flagbackup=True, mode='manual',antenna="81" ) #for HERA19
    flagdata(msname, flagbackup=True, mode='manual',spw="0:0~65" )#channels 0-65 of spw 0
    flagdata(msname, flagbackup=True, mode='manual',spw="0:377~387" )
    flagdata(msname, flagbackup=True, mode='manual',spw="0:850~854" )
    flagdata(msname, flagbackup=True, mode = 'manual' , spw = "0:930~1023")
    flagdata(msname, flagbackup=True, mode = 'manual' , spw = "0:831" )
    flagdata(msname, flagbackup=True, mode = 'manual' , spw = "0:769" )
    flagdata(msname, flagbackup=True, mode = 'manual' , spw = "0:511" )
    flagdata(msname, flagbackup=True, mode = 'manual' , spw = "0:913" )
    flagdata(msname, autocorr = True )

    return


def gen_image(msname,imagename): # Makes image
    clean(msname,imagename=imagename,niter =500,weighting = 'briggs',robust =0,imsize =[512 ,512] ,cell=['500 arcsec'] ,mode='mfs',nterms =1,spw='0:150~900',stokes='IQUV') #psf = point spread fuction. It describes the response of an imaging system to a point source or point objec

def mkinitmodel(msname,ext): #Model you give casa
    cl.addcomponent(flux =1.0 ,fluxunit='Jy', shape = 'point' ,dir='J2000 17h45m40.0409s -29d0m28.118s')
    cl.rename('GC'+ext+'.cl')
    cl.close()
    ft(msname , complist = 'GC'+ext+'.cl' , usescratch = True )

def mkinitmodel(msname,ext):
    direction = "J2000 02h00m12.7s -30d53m27s"
    ref_freq = "151MHz"
    
    # gleam 020012 -305327
    cl.addcomponent(label="GLEAM0200-3053", flux=17.5, fluxunit="Jy", dir=direction, freq=ref_freq,
                    shape="point", spectrumtype='spectral index', index=-0.86)
        
    # gleam 0150 -2931
    cl.addcomponent(label="GLEAM0150-2931", flux=16.6, fluxunit="Jy", dir="J2000 01h50m36s -29d31m59s", freq=ref_freq,shape="point", spectrumtype='spectral index', index=-0.80)
                    
    # gleam 015200 -294056
    cl.addcomponent(label="GLEAM0152-2940", flux=4.7, fluxunit="Jy", dir="J2000 01h52m00s -29d40m56s", freq=ref_freq,shape="point", spectrumtype='spectral index', index=-0.73)
                    
    # gleam 013411 -362913
    cl.addcomponent(label="GLEAM0134-3629", flux=18.5, fluxunit="Jy", dir="J2000 01h34m11.6s -36d29m13s", freq=ref_freq,shape="point", spectrumtype='spectral index', index=-0.72)
        
    cl.rename("gleam4comp."+ext+".cl")
    cl.close()
    ft(msname , complist = "gleam4comp."+ext+".cl" , usescratch = True )
    # save
    #if os.path.exists("gleam02.cl"):
    #    shutil.rmtree("gleam02.cl")
    #cl.rename("gleam02.cl")

    
def mkmodel(msname,ext): # PMN Field Source Model 
    # https://casa.nrao.edu/docs/UserMan/casa_cookbook006.html
    # https://casa.nrao.edu/docs/CasaRef/componentlist.addcomponent.html#x254-2540001.2.1
    fornaxAi =  'J2000 03h24m07.9s -37d16m25s' #direction measure of the source
    fornaxAii = 'J2000 03h21m37.9s -37d08m51s'
    pmn0324 = 'J2000 03h51m35.7s -27d44m35s'
    cl.addcomponent(shape = 'point', flux =35.0 ,fluxunit='Jy', freq='0.150GHz',dir=pmn0324) #add a component to component list (cl)
    cl.addcomponent(shape='point', flux=43, fluxunit='Jy', freq='0.150GHz', dir=fornaxAi) # non-SI unit of spectral flux density
    cl.addcomponent(shape='point', flux=86, fluxunit='Jy', freq='0.150GHz', dir=fornaxAii)
    cl.rename('PMN'+ext+'.cl')
    cl.close()
    ft(msname,complist='PMN'+ext+'.cl',usescratch=True)# write the model into the MODEL_DATA column of your data ('msname')


def clear_cal(msname):# Re-initializes the calibration for visibility dataset. It undoes applycal by restting MODEL_DATA & CORRECTED_DATA to DATA.
    clearcal(msname) #Imaging Task

def phscal(msname):#solver Task
    import os,sys
    kc = os.path.basename(msname)  + ".K.cal"
    bc = os.path.basename(msname)  + ".B.cal"
    gaincal(msname,caltable=kc,gaintype = 'K' , solint='inf',refant='10') # Delay calibration solution refantHERA19GC 10 refantFornax 0
    applycal(msname,gaintable=[kc])
    bandpass(msname,caltable=bc,solint='inf',combine='scan',refant='10') # scan and inf makes a single solution for entire dataset
    applycal(msname,gaintable=[bc])
    return

def phscal(msname):
    ra = '53'
    gaintables = []
    kc = os.path.basename(msname)  + ".K.cal"
    gpc = os.path.basename(msname)  + ".Gphs.cal"
    gac = os.path.basename(msname)  + ".Gamp.cal"
    bc = os.path.basename(msname)  + ".B.cal"

    gaincal(msname, caltable=kc, gaintype='K', solint='inf', refant=ra, gaintable=gaintables)
    gaintables.append(kc)

    gaincal(msname, caltable=gpc, gaintype='G', solint='inf', refant=ra, calmode='p', gaintable=gaintables)
    gaintables.append(gpc)

    gaincal(msname, caltable=gac, gaintype='G', solint='inf', refant=ra, calmode='a',gaintable=gaintables)
    gaintables.append(gac)

    bandpass(vis=msname,caltable=bc, gaintable=gaintables, solint='inf',combine='scan', refant=ra)
    gaintables.append(bc)
    
    applycal(msname, gaintable=gaintables)
    
    return


def phscal(msname):
    import os,sys
    
    kc = os.path.basename(msname)+".Kcal"
    bc = os.path.basename(msname)+".Bcal"
    pc = os.path.basename(msname)+".Dcal"
    gaincal(msname,caltable=kc,gaintype = 'K' , solint='inf',refant='10')
    applycal(msname,gaintable=[kc])
    bandpass(msname,caltable=bc,solint='inf',combine='scan',refant='10')
    applycal(msname,gaintable=[bc])
    polcal('trial3zen.2458115.24482.HH.uvR.MS', caltable=pc, field="", spw="", intent="", selectdata=True, timerange="", uvrange="", antenna="", scan="", observation="", msselect="", solint="inf", combine="scan", preavg=200, refant="", minblperant=4, minsnr=3.0, poltype="Dflls", smodel=[1,0,0,0], append=False, docallib=False, callib="")
    
#    polcal(msname, #I no longer use this one because I'm sure this is wrong.
    #       caltable=pc,
    #       solint='inf',
    #       combine='scan',
    #       preavg=200,
    #       poltype='Df',
    #        #       refant='10',
    #       smodel=[1,0,0,0],
    #       append=False,
#       minsnr=3.0)
    applycal(msname,gaintable=[pc])


#---------------------------------------

#preprocessing.py
def reorder(msname):
    import casac
    ms=casac.casac.table()
    ms.open(msname,nomodify=False)
    a1 , a2 , data = [ms.getcol(x) for x in [ "ANTENNA1" , "ANTENNA2" , "DATA" ]]
    m = a1 > a2
    data [: ,: ,m]= data [: ,: , m ]. conj ()
    x = a2 [ m ]
    a2 [ m ]= a1 [ m ]
    a1 [ m ]= x
    ms.putcol("ANTENNA1",a1)
    ms.putcol("ANTENNA2",a2)
    ms.putcol("DATA",data)
    ms.flush ()
    ms.close ()

def flag(msname):
    
    #bad for HERA 19: 22, 43, 80, 81
    #bad for HERA 47: 50, 98 for the smaller data set we have on folio
    #refantHERA19GC 10 refantFornax 0

    flagdata(msname, flagbackup=True, mode='manual',antenna="50" )
    # The more official-er HERA-19: 22, 43, , 80, 81
    #flagdata(msname, flagbackup=True, mode='manual',antenna="22" )
    #flagdata(msname, flagbackup=True, mode='manual',antenna="43" )
    #flagdata(msname, flagbackup=True, mode='manual',antenna="80" )
    #flagdata(msname, flagbackup=True, mode='manual',antenna="81" )
    # For flagging completely raw data
    flagdata(msname, flagbackup=True, mode='manual',spw="0:0~65" ) #channels 0-65 of spw 0
    flagdata(msname, flagbackup=True, mode='manual',spw="0:377~387" )
    flagdata(msname, flagbackup=True, mode='manual',spw="0:850~854" )
    flagdata(msname, flagbackup=True, mode = 'manual' , spw = "0:930~1023" )
    flagdata(msname, flagbackup=True, mode = 'manual' , spw = "0:831" )
    flagdata(msname, flagbackup=True, mode = 'manual' , spw = "0:769" )
    flagdata(msname, flagbackup=True, mode = 'manual' , spw = "0:511" )
    flagdata(msname, flagbackup=True, mode = 'manual' , spw = "0:913" )
    flagdata(msname, autocorr = True )
    
# Just do this once on the master copy
prefix = 'zen.2458050.50580.HH'
msfile=prefix+'.MS'
uvfitsfile = 'OriginalData/'+prefix+'.uvfits'
importuvfits(vis=msfile,fitsfile=uvfitsfile)
reorder(msfile)
flag(msfile)

#calibrate_ms.py
def gen_image(msname,imagename):# This reconstructs a model of the sky
    clean(msname,imagename=imagename,niter=500,weighting = 'briggs',robust =0,imsize =[512 ,512] ,cell=['500 arcsec'] ,mode='mfs',nterms =1,spw='0:150~900',stokes='IQUV')

def mkinitmodel(msname): #Create a Component List for Selfcal
    
    # for a point source with no spectral index
    cl.addcomponent(flux =35.0 ,fluxunit='Jy', shape = 'point' ,dir='J2000 03h51m35.7s -27d44m35s')
    
# for a Gaussian with a spectral index
    #cl.addcomponent(flux=1.25, fluxunit='mJy', polarization='Stokes',
    #dir='J2000 19h30m00s 15d00m00s', shape='gaussian', majoraxis='10arcsec',
    #minoraxis='6arcsec', positionangle='0deg', freq='1.25GHz',
    #spectrumtype='spectral index', index=-0.8)
    ###you can add more components if you wish by calling addcomponent repeatedly with different params

    ##save it to disk
    cl.rename('PMNJ0351-2744.cl')
    cl.close()
    
    ## write the model into the MODEL_DATA column of your data ('myms')
    ft(msname , complist = "PMNJ0351-2744.cl" , usescratch = True )

def clear_cal(msname):
    clearcal(msname)

def phscal(msname):
    import os,sys
    kc = os.path.basename(msname)  + ".K.cal"
    bc = os.path.basename(msname)  + ".B.cal"
    gaincal(msname,caltable=kc,gaintype = 'K' , solint='inf',refant='0')
    applycal(msname,gaintable=[kc])
    bandpass(msname,caltable=bc,solint='inf',combine='scan',refant='0')
    applycal(msname,gaintable=[bc])

gen_image(msname,msname+'.raw')

mkinitmodel(msname)
clearcal(msname)
phscal(msname)
imagename='zen.2458050.50580.HH.MS.phs.image'
fitsimage=imagename+'.fits'
exportfits()

#-----------------------------
def reorder(msname):
    import casac
    ms=casac.casac.table()
    ms.open(msname,nomodify=False)
    a1 , a2 , data = [ms.getcol(x) for x in [ "ANTENNA1" , "ANTENNA2" , "DATA" ]]
    m = a1 > a2
    data [: ,: ,m]= data [: ,: , m ]. conj ()
    x = a2 [ m ]
    a2 [ m ]= a1 [ m ]
    a1 [ m ]= x
    ms.putcol("ANTENNA1",a1)
    ms.putcol("ANTENNA2",a2)
    ms.putcol("DATA",data)
    ms.flush ()
    ms.close ()

def flag(msname):
    # The more official-er HERA-19: 22, 43, 80, 81
    inp
    flagdata(msname, flagbackup=True, mode='manual',antenna="43" )
    flagdata(msname, flagbackup=True, mode='manual',antenna="80" )
    flagdata(msname, flagbackup=True, mode='manual',antenna="81" )
    # For flagging completely raw data
    flagdata(msname, flagbackup=True, mode='manual',spw="0:0~65" )
    flagdata(msname, flagbackup=True, mode='manual',spw="0:377~387" )
    flagdata(msname, flagbackup=True, mode='manual',spw="0:850~854" )
    flagdata(msname, flagbackup=True, mode = 'manual' , spw = "0:930~1023" )
    flagdata(msname, flagbackup=True, mode = 'manual' , spw = "0:831" )
    flagdata(msname, flagbackup=True, mode = 'manual' , spw = "0:769" )
    flagdata(msname, flagbackup=True, mode = 'manual' , spw = "0:511" )
    flagdata(msname, flagbackup=True, mode = 'manual' , spw = "0:913" )
    flagdata(msname, autocorr = True )
    
def gen_image(msname,imagename):
    clean(msname,imagename=imagename,niter =500,weighting = 'briggs',robust =0,imsize =[512 ,512] ,cell=['500 arcsec'] ,mode='mfs',nterms =1,spw='0:150~900',stokes='IQUV')

def mkinitmodel(msname):
    cl.addcomponent(flux =[35.0,0.68,0,0] ,fluxunit='Jy', shape = 'point' ,dir='J2000 03h51m35.7s -27d44m35s',polarization='Stokes',freq='145MHz',spectrumtype='spectral index',index=-0.67)
    cl.rename('PMNJ0351-2744.cl')
    cl.close()
    ft(msname , complist = "PMNJ0351-2744.cl" , usescratch = True )

#def mkpolznmodel(msname):
    
def clear_cal(msname):
    #it will set the MODEL_DATA column (if present) to unity in total intensity and zero in polarization, and it will set the CORRECTED_DATA column to the original (observed) DATA in the DATA column
    clearcal(msname)

def phscal(msname):
    import os,sys
    kc = os.path.basename(msname)  + ".K.cal"
    bc = os.path.basename(msname)  + ".B.cal"
    gaincal(msname,caltable=kc,gaintype = 'K' , solint='inf',refant='0')
    applycal(msname,gaintable=[kc])
    bandpass(msname,caltable=bc,solint='inf',combine='scan',refant='0')
    applycal(msname,gaintable=[bc])

# The blcal task determines baseline-based time- and/or frequency-dependent gains for all baselines in the data set.  Such solutions are in contrast to gaincal and bandpass solutions which are antenna-based and better constrained.
blcal(vis='data.ms',
      caltable='cal.M',                        # Output table name
      field='2',                               # A field with a very good model
      solint='inf',                            # single solution per baseline, spw
      gaintable=['cal.B','cal.gc','cal.G90s'], # all prior cal
      freqdep=False)                           # frequency-independent solution
#%% CREATE IMAGE
MSfilelist = glob.glob('*.MS')
name=MSfilelist
imgnam1='clean1_'
imgnam2='clean2_'
imgnam3='clean3_'

nwms=[]
for ms in MSfilelist:
    msf=ms.strip('zen.'+'HH.uv.MS')
    nwms.append(msf)#+'.R')


#1) K calibration: per-antenna delay calibration with the gaincal(..., gaintype=‘K’) task
#2) G calibration: per-antenna mean-phase calibration with the gaincal(..., gaintype=‘G’, calmode=‘p’) task
#3) A calibration: per-antenna mean-amp calibration with the gaincal(..., gaintype=‘G’, calmode=‘a’) task
#4) BP calibration: complex per-antenna bandpass calibration with the bandpass(...) task
i=0
for i in np.arange(len(MSfilelist)): #This is the longest part
    reorrder(msname=name[i])# Reorder antenna 1 & 2 in each correlation so 2 is greater than 1
    flag(msname=name[i])# Flagging bad frequency channels or antennas and autoccorelations in CASA
    gen_image(msname=name[i],imagename=imgnam1+nwms[i])# Imaging
    #mkinitmodel(msname=name[i],ext=nwms[i])# Calibration:Assume a calibrator to generate a model spectrum.We are going to use the Galactic Center in our case.
    mkmodel(msname=name[i],ext=nwms[i])
    clear_cal(msname=name[i])# Calibration:Solving for delays and gain solutions
    phscal(msname=name[i])
    gen_image(msname=name[i],imagename=imgnam2+nwms[i])# Imaging again
#clean(msname,imagename=imagename,niter =500,weighting = 'briggs',robust =0,imsize =[512 ,512] ,cell=['500 arcsec'] ,mode='mfs',nterms =1,spw='0:150~900',stokes='IQUV')

clean('zen.2458050.49835.HH.uv.MS','test_interactive1',niter=0,weighting = 'briggs',robust =0
      ,imsize =[512 ,512] ,pbcor=False, cell=['500 arcsec'] ,mode='mfs',nterms =1,spw='0:150~900',stokes='IQUV'
      ,interactive=False,npercycle=5,threshold='0.1mJy/beam')#,usescratch=True)

"""
project.[cfg].skymodel.flat.regrid.conv = input sky regridded to match
    the output image, and convolved with the output clean beam
    
    project.[cfg].image = synthesized image
    project.[cfg].flux.pbcoverage = primary beam correction for mosaic image
    project.[cfg].residual = residual image after cleaning
    project.[cfg].clean.last = parameter file of what parameters were used in
        the clean task
    project.[cfg].psf = synthesized (dirty) beam calculated from weighted uv
        distribution
project.[cfg].image.png = diagnostic figure of clean image and residual

project.[cfg].fidelity = fidelity image
    project.[cfg].analysis.png = diagnostic figure of difference and fidelity
    project.[cfg].simanalyze.last = saved input parameters for simanalyze task
    
    imagename.image – final cleaned (or dirty if niter=0 image)
    imagename.psf – the point spread function of the beam, useful to
    check whether image artifacts are due to poor psf
    imagename.model – an image of the clean components
    imagename.residual – the residual image after subtracting clean
    components, useful to check if more cleaning is needed
    imagename.flux – the primary beam response function. Relative sky sesitivity over a field – used to make
    a “flux correct image”, otherwise flux is only correct at the phase
    center(s). pbcor=T divides the .image by the .flux. Such images don’t
    look pretty because the noise at the edges are also increased, but
    flux densities should ONLY be calculated from pbcor’ed images.
    
"""
#%% CREATE .NPZ FILE IN CASA

tb.open('trial3zen.2458115.24482.HH.uvR.MS.Bcal');gains=tb.getcol('CPARAM');np.savez('trial3zen.2458115.24482.HH.uvR.MS.Bcal.npz',gains=gains)

MSBCALlist=glob.glob('*MSB.cal')

for msblist in MSBCALlist:
    tb.open(msblist)
    gain = tb.getcol('CPARAM') # use 'FPARAM' when using K.cal files
    np.savez(nwms[i]+'.npz',gains=gain)

d = np.load(nwms[i]+'.npz') #In python
gain=d['gains']
d.keys()

plt.imshow(np.abs(gain[0,:,:]).T, aspect='auto', interpolation='nearest');plt.colorbar();plt.show()
plt.plot(np.abs(gain[0,:,0]));plt.show()

fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(15,8))

axes[0].set_title('2458115.24482.HH.uvOCR: X Gain Solutions')
axes[0].imshow(np.abs(gain[0,:,:]).T, aspect='auto', interpolation='nearest')
fig.colorbar(axes[0].imshow(np.abs(gain[0,:,:]).T, aspect='auto', interpolation='nearest'),orientation='vertical',ax=axes[0])
axes[0].set_aspect('auto')

axes[1].set_title('2458115.24482.HH.uvOCR: Y Gain Solutions')
axes[1].imshow(np.abs(gain[1,:,:]).T, aspect='auto', interpolation='nearest')
fig.colorbar(axes[1].imshow(np.abs(gain[1,:,:]).T, aspect='auto', interpolation='nearest'),orientation='vertical',ax=axes[1])
axes[1].set_aspect('auto')

#fig.tight_layout()
fig.suptitle('Bandpass Calibration Solutions')
axes[0].set_ylabel('Antenna Number')
axes[0].set_xlabel('Channel')
axes[1].set_xlabel('Channel')
fig.savefig('2458115.24482.HH.uvOCR_BP.png')
plt.show()

#%%
plotants(vis='zen.2458050.49835.HH.uv.MS',figfile='zen.2458050.49835_antenna_layout.png')
plotms('zen.2458050.49835.HH.uv.MS') #very useful
browsetable('zen.2458050.49835.HH.uv.MS')
listvis('zen.2458050.49835.HH.uv.MS')
listobs('zen.2458050.49835.HH.uv.MS',selectdata=True,spw="",field="", #This may not produce anything
        antenna="",uvrange="",timerange="",correlation="",scan="",
        intent="",feed="",array="",observation="",verbose=True,
        listfile="",listunfl=False)
plotuv('zen.2458050.49835.HH.uv.MS')

vishead('zen.2458050.49835.HH.uv.MS',mode='list',listitems=[]) # very useful to get information about the MS
listcal(vis="zen.2455819.68380.uvcRRECAaU.MS", caltable="bandpass_per_channel_per_spw.MSB.cal")

# In CASA
plotcal(caltable= '3c391_ctm_mosaic_10s_spw0.B0',poln='R',
        xaxis='chan',yaxis='amp',field= 'J1331+3030',subplot=221,
        iteration='antenna',figfile='plotcal_3c391-3C286-B0-R-amp.png')
#
plotcal(caltable= '3c391_ctm_mosaic_10s_spw0.B0',poln='L',
        xaxis='chan',yaxis='amp',field= 'J1331+3030',subplot=221,
        iteration='antenna',figfile='plotcal_3c391-3C286-B0-L-amp.png')
#
plotcal(caltable= '3c391_ctm_mosaic_10s_spw0.B0',poln='R',
        xaxis='chan',yaxis='phase',field= 'J1331+3030',subplot=221,
        iteration='antenna',plotrange=[-1,-1,-180,180],
        figfile='plotcal_3c391-3C286-B0-R-phase.png')
#
plotcal(caltable= '3c391_ctm_mosaic_10s_spw0.B0',poln='L',
        xaxis='chan',yaxis='phase',field= 'J1331+3030',subplot=221,
        iteration='antenna',plotrange=[-1,-1,-180,180],
        figfile='plotcal_3c391-3C286-B0-L-phase.png')
"""
’amp’ — amplitude,
’phase’ — phase,
’real’ – the real part,
’imag’ — the imaginary part,
’snr’ – the signal-to-noise ratio,
iteration = ’antenna, time, field, spw’
"""

# You can export .image file as a .fits file then read fits file into python normally
exportfits('imagename.image','fitsimagename.fits') # Done in CASA for a single file

imagename = glob.glob('clean2*.image')
for im in imagename:
    ext = '.fits'
    fitsimagename = im.strip('.image')
    exportfits(imagename,fitsimagename+ext)
fitsimage = glob.glob(clean2*.fits)


# This section is to be done in python
test=fits.open('new_name.fits')
test.info()
test[0].header


...: import sys
...: from time import sleep
...: try:
...:     shell = sys.stdout
...: except:
...:     print('Run It In Shell')
...: dots = '........';
...: shell.write('Downloading')
...: sleep(0.5)
...: for dot in dots:
...:     shell.write(dot)
...:     sleep(0.1)
...: shell.write('\n')
...: sleep(0.4)
...: shell.write('Saving Files')
...: sleep(0.5)
...: for doot in dots:
...:     shell.write(dot)
...:     sleep(0.1)
...: shell.write('\n')
...: sleep(0.4)


