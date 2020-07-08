#!/bin/sh

#  heavy_flagging_polcal_casa.py
#  
#
#  Created by Tashalee Billings on 4/11/18.
#  

import numpy as np
import os,sys
from recipes.almapolhelpers import *

"""
Helpful links

https://casaguides.nrao.edu/index.php/3C286_Band6Pol_Calibration_for_CASA_4.3#label-comparison
https://casa.nrao.edu/casadocs-devel/stable/synthesis-calibration/instrumental-polarization-calibration
"""

msname = "NAME.MS"

kc = os.path.basename(msname)+".Kcal" #FPARMA
bc = os.path.basename(msname)+".Bcal" #CPARAM
gc = os.path.basename(msname)+".Gcal" #CPARAM
kcross = os.path.basename(msname)+".XDELcal" #cross-hand delay #FPARMA
xy0amb = os.path.basename(msname)+".xy0amb" # ambiguity #CPARAM
xy0 = os.path.basename(msname)+".xy0" #CPARAM
gc1 = os.path.basename(msname)+".Gcal1" #CPARAM
pc = os.path.basename(msname)+".Dcal" #CPARAM

gaincal(msname, #prints nothing
        caltable=kc,
        gaintype = 'K',
        solint='inf',
        refant='10') #hera19 10
bandpass(msname, #prints "solutions flagged due to...."
         caltable=bc,
         solint='inf',
         combine='scan',
         refant='10') #hera19 10
gaincal(vis=msname, #prints "solutions flagged due to...."
        caltable=gc,
        field='',                 # the instrumental polarization calibrator
        solint='int',
        smodel=[1,0,0,0],          # assume zero polarization
        gaintype='G',
        gaintable=[bc],
        parang=True)                  # so source poln properly rotated
gaincal(vis=msname, #prints nothing
        caltable=kcross,
        field='',
        solint='inf',
        combine='scan',
        refant='10', #hera19 10
        smodel=[1,0,0,0],
        gaintype='KCROSS',
        gaintable=[gc,bc])
qu=qufromgain(gc) # solves for stokes Q,U from previous gain calibrations #prints off a bunch of polarization parameters
gaincal(vis=msname, #prints XY phase, fractional polarization, net instramental (over baselines) polarization
        caltable=xy0amb,  # possibly with 180deg ambiguity
        field='',                 # the calibrator
        solint='inf',
        combine='scan',
        preavg=200.0,              # minimal parang change
        smodel=[1,0,0,0],          # non-zero U assumed
        gaintype='XYf+QU',
        gaintable=[gc,bc,kcross])  # all prior calibration
S=xyamb(xy0amb,qu=qu[0],xyout=xy0) #prints more polarization params and returns a new stokes vector
gaincal(vis=msname, #prints "solutions flagged due to...."
        caltable=gc1,
        field='',
        solint='int',
        smodel=S,                  # obtained from xyamb
        gaintype='G',
        gaintable=[bc],
        parang=True)                  # so source poln properly rotated
# flag the data here
polcal(vis=msname,caltable=pc,field='',spw='',intent='',selectdata=True,timerange='',uvrange='',antenna='',scan='',observation='',msselect='',solint='inf',combine='scan',preavg=200,refant='',minblperant=4,minsnr=3.0,poltype='Dflls',smodel=[1,0,0,0], append=False, docallib=False,callib='',gaintable=[gc1, bc, xy0], gainfield=[''], interp=[], spwmap=[])

# Apply all of the calibrations
applycal(vis=msname, field='', spw='', intent='', selectdata=True, timerange='', uvrange='', antenna='',scan='', observation='', msselect='', docallib=False, callib='', gaintable=[kc, gc1, bc, kcross, pc], gainfield=[], interp=[], spwmap=[], calwt=[True], parang=False, applymode='', flagbackup=True)
#flag the data here

fileC=[bc,gc,xy0amb,xy0,gc1,pc]
fileF=[kc,kcross]

for fc in fileC:
    tb.open(fc)
    gains=tb.getcol('CPARAM')
    np.savez(fc+'.npz',gains=gains)
for ff in fileF:
    tb.open(ff)
    gains=tb.getcol('FPARAM')
    np.savez(ff+'.npz',gains=gains)
