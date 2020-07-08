#!/bin/sh

#  KGB_calibration.py
#  
#
#  Created by Tashalee Billings on 5/19/18.
#  

import numpy as np
import argparse
import shutil
import os, glob, sys

# Preliminary Calibration Steps

def reorder(namems):
    import casac
    ms=casac.casac.table()
    ms.open(namems,nomodify=False)
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

def flag(namems): #You might have to update this. Flag MS or calibration Tables.
    flagdata(namems, flagbackup=True, mode='manual',antenna="22" ) #for HERA19
    flagdata(namems, flagbackup=True, mode='manual',antenna="43" ) #for HERA19
    flagdata(namems, flagbackup=True, mode='manual',antenna="80" ) #for HERA19
    flagdata(namems, flagbackup=True, mode='manual',antenna="81" ) #for HERA19
    
    flagdata(namems, flagbackup=True, mode='manual',spw="0:0~65" )#channels 0-65 of spw 0
    #flagdata(namems, flagbackup=True, mode='manual',spw="0:377~387" )
    #flagdata(namems, flagbackup=True, mode='manual',spw="0:850~854" )
    flagdata(namems, flagbackup=True, mode = 'manual' , spw = "0:930~1023")
    #flagdata(namems, flagbackup=True, mode = 'manual' , spw = "0:831" )
    #flagdata(namems, flagbackup=True, mode = 'manual' , spw = "0:769" )
    #flagdata(namems, flagbackup=True, mode = 'manual' , spw = "0:511" )
    #flagdata(namems, flagbackup=True, mode = 'manual' , spw = "0:913" )
    
    flagdata(namems, autocorr = True )

def initialmodel(namems):
    
    #Galactic Center Model you give casa
    cl.addcomponent(flux =1.0 ,fluxunit='Jy', shape = 'point' ,dir='J2000 17h45m40.0409s -29d0m28.118s')
    
    if os.path.exists("GC.cl"):
        shutil.rmtree("GC.cl")
    cl.rename('GC.cl')
    cl.close()
    ft(namems , complist = 'GC.cl' , usescratch = True )

#Convert uvfits to MS
execfile("uvfits2ms.py")
print("Executed uvfits2ms.py")

parser = argparse.ArgumentParser(description='Perform Delay, Phase, and Bandpass Calibration. Run with casa as: casa -c KGB_calibration.py --<args>=')

parser.add_argument("-c", type=str, help="Name of this script.")
parser.add_argument("--refant", type=str, default=None, help="The reference Antenna used to calibrate.")
parser.add_argument("--calmode", type=str,default=None, choices=['p', 'a', 'ap'], help="Calibration mode for GAINCAL type G. Can be 'p','a','ap'.")
parser.add_argument("--ms", action='append', dest='measurement_set',type=str ,#default=[],
               help="Name of the measurement set to be calibrated. 'zen.JD2000.uv.MS'")

# USER$ casa --nologger --nocrashreport --nogui --agg -c KGB_calibration.py --refant=10 --calmode='p' --ms='name' --ms='hmm'
#if you want the positional argument to be optional then write '--NameOfArg'.

# https://pymotw.com/2/argparse/ and https://stackoverflow.com/questions/15753701/argparse-option-for-passing-a-list-as-option(Helpful websites)

print("Created Positional Argument.")

args = parser.parse_args()

if __name__ == '__main__': #This only runs is we run directly but not if you try to import it.

    msname = args.measurement_set
    print args.measurement_set

    if len(msname)==1:
    
        msname = args.measurement_set[0]
        print msname

        reorder(msname)
        flag(msname)
        initialmodel(msname)
        print("Finished preliminary calibrations.")

        kc = os.path.basename(msname)  + ".K.cal"
        gc = os.path.basename(msname)  + ".G.cal"
        bc = os.path.basename(msname)  + ".B.cal"

        gaincal(msname,caltable=kc,gaintype = 'K' , solint='inf',refant=args.refant)
        applycal(msname,gaintable=[kc])
        print("Created "+kc)
        gaincal(msname,caltable=gc,gaintype = 'G' , solint='int',refant=args.refant,calmode=args.calmode)
        applycal(msname,gaintable=[gc])
        print("Created "+gc)
        bandpass(msname,caltable=bc,solint='inf',combine='scan',refant=args.refant)
        applycal(msname,gaintable=[bc])
        print("Created "+bc)

    else:
        for name in msname:
            name = "'"+ str(name)+"'"
            print("Calibrating "+name)
            
            reorder(name)
            flag(name)
            initialmodel(name)
            print("Finished preliminary calibrations.")
            
            kc = os.path.basename(name)  + ".K.cal"
            gc = os.path.basename(name)  + ".G.cal"
            bc = os.path.basename(name)  + ".B.cal"

            #gaincal(name,caltable=kc,gaintype = 'K' , solint='inf',refant=args.refant)
            #applycal(name,gaintable=[kc])
            print("Created "+kc)
            #gaincal(name,caltable=gc,gaintype = 'G' , solint='int',refant=args.refant,calmode=args.calmode)
            #applycal(name,gaintable=[gc])
            print("Created "+gc)
            #bandpass(name,caltable=bc,solint='inf',combine='scan',refant=args.refant)
            #applycal(name,gaintable=[bc])
            print("Created "+bc)

