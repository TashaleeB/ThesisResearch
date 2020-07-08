#!/bin/sh

#  calibration_tests.py
#
#
#  Created by Tashalee Billings on 6/7/18.
#

"""
   These command are for CASA for ONLY .uv files. Before that you need to write a short script in the commandline that runs "merge_4pol.py."
   
import glob,os

JD = "*"
list = glob.glob("zen."+JD+".xx.HH.uv")

list_prefix=[]
list_suffix=[]
for l in list:
    list_prefix.append(l[:-9]) #remove last 9 characters
    list_suffix.append(l[-6:]) #keeps the last 6 characters

for i in range(len(list_prefix)):
    command = "python merge_4pols.py --prefix="+list_prefix[i]+" --suffix="+list_suffix[i]
    os.system(command)
"""

# Location of the two types of simulated data on folio2
# /data4/paper/SimulatedData/HERA19_GSM2008_NicCST2016_full_day/D_term_corrupted/
# /data4/paper/SimulatedData/HERA19_GSM2008_NicCST2016_full_day/true_visibilities/

import numpy as np
import time
import argparse
import os, glob, sys
#import itertools as iter

#-----------------------------------------------
# Preliminary Calibration Steps
#-----------------------------------------------

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
    import shutil
    import numpy as np
    
    #Galactic Center Model you give casa
#    cl.addcomponent(flux =1.0 ,fluxunit='Jy', shape = 'point' ,dir='J2000 17h45m40.0409s -29d0m28.118s')

    ref_freq = "408MHz"
    alpha = -0.5
    
    cl.addcomponent(flux=3709, fluxunit="Jy", shape = "point",
                    spectrumtype="spectral index", freq=ref_freq,
                    index=alpha, #The spectral index α such that flux density S as a function of frequency ν is given by the formula: S ∝ ν^α
                    dir="J2000 17h45m40.0409s -29d0m28.118s")
    
    if os.path.exists("GC.cl"):
        shutil.rmtree("GC.cl")
        cl.rename('GC.cl')
        cl.close()
        ft(namems , complist = 'GC.cl' , usescratch = True )
    
#-----------------------------------------------
#Convert uvfits to MS
#-----------------------------------------------
execfile("uvfits2ms.py")
print("Executed uvfits2ms.py")

og_ms = glob.glob("*.MS")

parser = argparse.ArgumentParser(description='Perform Delay, Phase, and Bandpass Calibration. Run with casa as: casa -c KGB_calibration.py --<args>=')

parser.add_argument("-c", type=str, help="Calibrationg for different permutations.")
parser.add_argument("--refant", type=str, default=None, help="The reference Antenna used to calibrate.")
parser.add_argument("--nocal", type=str, default='False', choices=['True', 'False'], help="Are you imaging data that either is perfectly calibrated OR simulated data that does not require CLEAN iterations, niter=0?")
parser.add_argument("--calmode_G", type=str,default=None, choices=['p', 'a', 'ap'], help="Calibration mode for GAINCAL type G. Can be 'p','a','ap'.")
parser.add_argument("--calmode_pol", type=str,default='Df', choices=['Df','Dflls'], help="Calibration mode for Polarized Calibration. Can be 'Df' or 'Dflls'. The default is set to 'Df'. 'Dflls' - A specialized mode which explicitly solves a linearized form of the cross-hand data for the D-terms. 'Df' - Solve for instrumental polarization (leakage D-terms), using the transform of a specified IQU model; requires no parallactic angle coverage, but if the source polarization is non-zero, the gain calibration must have the correct R-L phase registration. (Note: this is unlikely, so just use 'D+X' to let the cross-hand phase registration float). This will produce a calibration table of type D.")

#parser.add_argument("--ms", action='append', dest='measurement_set',type=str ,#default=[],help="Name of the measurement set to be calibrated. 'zen.JD2000.uv.MS'")
# EXAMPLE:
#   command_line = "casa --nologger --nocrashreport --nogui --agg -c ../../calibration_tests_in_CASA.py --refant=10 --nocal='False' --calmode_G='p'"

print("Created Positional Argument.")

args = parser.parse_args()

if __name__ == '__main__': #This only runs is we run directly but not if you try to import it.

    #msname_ = args.measurement_set
    msname_ = glob.glob("*.MS")
    nocalib = args.nocal
    
    for msname in msname_:
    
        #msname = args.measurement_set[0]
        print("The measurement set you are working with is ... "+msname)
        print("No Calibration Needed Parameter is set to "+nocalib)
        time.sleep(2)
        
        print("Preparing to perform preliminary calibration")
        time.sleep(2)
        reorder(msname)
        flag(msname)
        initialmodel(msname)
        print("Finished preliminary calibrations.")
        
        # Applying calibrations
        if nocalib == 'False':
            
            #-----------------------------------------------
            # Make copies of preliminary calibrated MS so that we can apply different calibration permiatations.
            #-----------------------------------------------
            cal_exten = ['K','G','B','KB','BK','KG','GK','BG','GB','KBG','KGB','BKG','BGK','GKB','GBK']
            cal_exten_ = ['K','G','B','KB','KG','BG','GB','KBG','KGB'] #What we want
            list_ms=[]
            #list_ms=glob.glob("*.MS")[1:]
        
            #Makes copies of orginial MS
            for cal in cal_exten_:
                os.system("scp -r "+msname+" "+msname[:-2]+cal+".MS")
                msn = msname[:-2]+cal+".MS"
                list_ms.append(msn)
                print(msn)
                
            #x = [1,2,3]
            #print(list(itertools.permutations(x,2)))
            #>>> [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]
            
            print("Starting the CLEAN loop with number of iterations set to 0 with no Calibrations applied.")
            # Loop through CLEAN task
            print "Start Time: %s" % time.ctime()
            ms = list_ms[0] #this should be picking *.K.MS file
            clean(ms,ms[:-5]+'_no_cal_niter0',niter=0, weighting = 'briggs',robust =0, imsize =[512 ,512] ,pbcor=False, cell=['500 arcsec'] ,mode='mfs',nterms=1, spw='0:150~900', stokes='IQUV',interactive=False, npercycle=5, threshold='0.1mJy/beam')
            print "End Time: %s" % time.ctime()
            
            print("Beginning The Calibration Process.")
            
            print("Start Time : %s" % time.ctime())
            #for i in range(len(cal_exten)):
            for m in list_ms:
                if m[-5:]=="."+cal_exten[0]+".MS":#K if .K.cal
                    calitable = os.path.basename(m)  + "."+cal_exten[0]+".cal"
                    gaincal(m,caltable=calitable,gaintype = 'K' , solint='inf',refant=args.refant)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                if m[-5:]=="."+cal_exten[1]+".MS":#G if .G.cal
                    calitable = os.path.basename(m)  + "."+cal_exten[1]+".cal"
                    gaincal(m,caltable=calitable,gaintype = 'G', solint='int', refant=args.refant, calmode=args.calmode_G)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                if m[-5:]=="."+cal_exten[2]+".MS":#B if .B.cal
                    calitable = os.path.basename(m)  + "."+cal_exten[2]+".cal"
                    bandpass(m,caltable=calitable,solint='inf',combine='scan',refant=args.refant)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                if m[-6:]=="."+cal_exten[3]+".MS":#KB if .KB.cal
                    calitable = os.path.basename(m)  + "."+cal_exten[0]+".cal"
                    gaincal(m,caltable=calitable,gaintype = 'K' , solint='inf',refant=args.refant)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                    calitable = os.path.basename(m)  + "."+cal_exten[2]+".cal"
                    bandpass(m,caltable=calitable,solint='inf',combine='scan',refant=args.refant)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                if m[-6:]=="."+cal_exten[4]+".MS":#BK if .BK.cal
                    calitable = os.path.basename(m)  + "."+cal_exten[2]+".cal"
                    bandpass(m,caltable=calitable,solint='inf',combine='scan',refant=args.refant)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                    calitable = os.path.basename(m)  + "."+cal_exten[0]+".cal"
                    gaincal(m,caltable=calitable,gaintype = 'K' , solint='inf',refant=args.refant)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                if m[-6:]=="."+cal_exten[5]+".MS":#KG if .KG.cal
                    calitable = os.path.basename(m)  + "."+cal_exten[0]+".cal"
                    gaincal(m,caltable=calitable,gaintype = 'K' , solint='inf',refant=args.refant)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                    calitable = os.path.basename(m)  + "."+cal_exten[1]+".cal"
                    gaincal(m,caltable=calitable,gaintype = 'G', solint='int', refant=args.refant, calmode=args.calmode_G)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                if m[-6:]=="."+cal_exten[6]+".MS":#GK if .GK.cal
                    calitable = os.path.basename(m)  + "."+cal_exten[1]+".cal"
                    gaincal(m,caltable=calitable,gaintype = 'G', solint='int', refant=args.refant, calmode=args.calmode_G)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                    calitable = os.path.basename(m)  + "."+cal_exten[0]+".cal"
                    gaincal(m,caltable=calitable,gaintype = 'K' , solint='inf',refant=args.refant)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                if m[-6:]=="."+cal_exten[7]+".MS":#BG if .BG.cal
                    calitable = os.path.basename(m)  + "."+cal_exten[2]+".cal"
                    bandpass(m,caltable=calitable,solint='inf',combine='scan',refant=args.refant)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                    calitable = os.path.basename(m)  + "."+cal_exten[1]+".cal"
                    gaincal(m,caltable=calitable,gaintype = 'G', solint='int', refant=args.refant, calmode=args.calmode_G)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                if m[-6:]=="."+cal_exten[8]+".MS":#GB if .GB.cal
                    calitable = os.path.basename(m)  + "."+cal_exten[1]+".cal"
                    gaincal(m,caltable=calitable,gaintype = 'G', solint='int', refant=args.refant, calmode=args.calmode_G)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                    calitable = os.path.basename(m)  + "."+cal_exten[2]+".cal"
                    bandpass(m,caltable=calitable,solint='inf',combine='scan',refant=args.refant)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                if m[-7:]=="."+cal_exten[9]+".MS":#KBG if .KBG.cal
                    calitable = os.path.basename(m)  + "."+cal_exten[0]+".cal"
                    gaincal(m,caltable=calitable,gaintype = 'K' , solint='inf',refant=args.refant)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                    calitable = os.path.basename(m)  + "."+cal_exten[2]+".cal"
                    bandpass(m,caltable=calitable,solint='inf',combine='scan',refant=args.refant)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                    calitable = os.path.basename(m)  + "."+cal_exten[1]+".cal"
                    gaincal(m,caltable=calitable,gaintype = 'G', solint='int', refant=args.refant, calmode=args.calmode_G)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                if m[-7:]=="."+cal_exten[10]+".MS":#KGB if .KGB.cal
                    calitable = os.path.basename(m)  + "."+cal_exten[0]+".cal"
                    gaincal(m,caltable=calitable,gaintype = 'K' , solint='inf',refant=args.refant)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                    calitable = os.path.basename(m)  + "."+cal_exten[1]+".cal"
                    gaincal(m,caltable=calitable,gaintype = 'G', solint='int', refant=args.refant, calmode=args.calmode_G)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                    calitable = os.path.basename(m)  + "."+cal_exten[2]+".cal"
                    bandpass(m,caltable=calitable,solint='inf',combine='scan',refant=args.refant)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                if m[-7:]=="."+cal_exten[11]+".MS":#BKG if .BKG.cal
                    calitable = os.path.basename(m)  + "."+cal_exten[2]+".cal"
                    bandpass(m,caltable=calitable,solint='inf',combine='scan',refant=args.refant)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                    calitable = os.path.basename(m)  + "."+cal_exten[0]+".cal"
                    gaincal(m,caltable=calitable,gaintype = 'K' , solint='inf',refant=args.refant)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                    calitable = os.path.basename(m)  + "."+cal_exten[1]+".cal"
                    gaincal(m,caltable=calitable,gaintype = 'G', solint='int', refant=args.refant, calmode=args.calmode_G)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                if m[-7:]=="."+cal_exten[12]+".MS":#BGK if .BGK.cal
                    calitable = os.path.basename(m)  + "."+cal_exten[2]+".cal"
                    bandpass(m,caltable=calitable,solint='inf',combine='scan',refant=args.refant)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                    calitable = os.path.basename(m)  + "."+cal_exten[1]+".cal"
                    gaincal(m,caltable=calitable,gaintype = 'G', solint='int', refant=args.refant, calmode=args.calmode_G)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                    calitable = os.path.basename(m)  + "."+cal_exten[0]+".cal"
                    gaincal(m,caltable=calitable,gaintype = 'K' , solint='inf',refant=args.refant)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                if m[-7:]=="."+cal_exten[13]+".MS":#GKB if .GKB.cal
                    calitable = os.path.basename(m)  + "."+cal_exten[1]+".cal"
                    gaincal(m,caltable=calitable,gaintype = 'G', solint='int', refant=args.refant, calmode=args.calmode_G)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                    calitable = os.path.basename(m)  + "."+cal_exten[0]+".cal"
                    gaincal(m,caltable=calitable,gaintype = 'K' , solint='inf',refant=args.refant)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                    calitable = os.path.basename(m)  + "."+cal_exten[2]+".cal"
                    bandpass(m,caltable=calitable,solint='inf',combine='scan',refant=args.refant)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                if m[-7:]=="."+cal_exten[14]+".MS":#GBK if .GBK.cal
                    calitable = os.path.basename(m)  + "."+cal_exten[1]+".cal"
                    gaincal(m,caltable=calitable,gaintype = 'G', solint='int', refant=args.refant, calmode=args.calmode_G)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                    calitable = os.path.basename(m)  + "."+cal_exten[2]+".cal"
                    bandpass(m,caltable=calitable,solint='inf',combine='scan',refant=args.refant)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                    calitable = os.path.basename(m)  + "."+cal_exten[0]+".cal"
                    gaincal(m,caltable=calitable,gaintype = 'K' , solint='inf',refant=args.refant)
                    print("Created "+glob.glob(calitable)[0])
                    applycal(m,gaintable=[calitable])
                print "End Time: %s" % time.ctime()
                time.sleep(3)

            print("Starting the CLEAN loop with number of iterations set to 500.")
                # Loop through CLEAN task
            print "Start Time: %s" % time.ctime()
            for ms in list_ms:
                clean(ms,ms[:-3],niter=500, weighting = 'briggs',robust =0, imsize =[512 ,512] ,pbcor=False, cell=['500 arcsec'] ,mode='mfs',nterms=1, spw='0:150~900', stokes='IQUV',interactive=True, npercycle=5, threshold='0.1mJy/beam')
            print "End Time: %s" % time.ctime()

        else:
            print("Starting the CLEAN loop with number of iterations set to 0.")
            print "Start Time: %s" % time.ctime()
            for ms in list_ms:
                clean(ms,ms[:-3],niter=0, weighting = 'briggs',robust =0, imsize =[512 ,512] ,pbcor=False, cell=['500 arcsec'] ,mode='mfs',nterms=1, spw='0:150~900', stokes='IQUV',interactive=False, npercycle=5, threshold='0.1mJy/beam')
                images_file=glob.glob(ms[:-3]+".imgage")[0]
                print(images_file)
            print "End Time: %s" % time.ctime()

        time.sleep(5)
        print("Converting IMAGE file to fits image.")
        list_image = glob.glob("*.image")
        for im in list_image:
            exportfits(im,im+".fits")
            print(im+".fits")

        #-----------------------------------------------
        # Convert Calibration Solutions to NPZ files : Done in CASA
        #-----------------------------------------------

        time.sleep(5)
        print("Convert Calibration files to NPZ file.")

        cparamB = glob.glob("*.B.cal")
        cparamG = glob.glob("*.G.cal")
        fparamK = glob.glob("*.K.cal")

        for c in cparamB:
            tb.open(c)
            gain = tb.getcol('CPARAM')
            np.savez(c+'.npz',gains=gain)
            print("Created "+ c+'.npz')

        for c in cparamG:
            tb.open(c)
            gain = tb.getcol('CPARAM')
            np.savez(c+'.npz',gains=gain)
            print("Created "+ c+'.npz')

        for f in fparamK:
            tb.open(f)
            gain = tb.getcol('FPARAM')
            np.savez(f+'.npz',gains=gain)
            print("Created "+ f+'.npz')
        #del(c)
        #del(f)
