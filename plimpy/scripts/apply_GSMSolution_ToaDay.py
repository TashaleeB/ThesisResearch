#!/bin/sh

#  apply_GSMSolution_ToaDay.py
#  
#
#  Created by Tashalee Billings on 12/31/18.
#  

import glob,os

bp_scaled_calfits = '/data4/paper/HERA19Golden/RawData/2457548/?????'
apply_abscal = '/data4/paper/HERA19Golden/kohn18_analysis/apply_abscal.sh'
miriad_list = sorted(glob.glob("/data4/paper/HERA19Golden/RawData/2457548/*.HH.uvcRP"))

# Make new directory to put GSM Abscal'd Data
abscal_dir = '/data4/paper/HERA19Golden/CalibratedData/2457548/GSM_AbscalData/'
os.system("mkdir " +abscal_dir)
#os.path.dirname(mf) #returns the head of the path
#os.path.basename(mf) #returns the tail of the path

# Loop through file by file for a single day and apply an GSM-model solutions to them and save file to new directory
for mf in miriad_list:
    new_mf_path = os.path.join(abscal_dir,(os.path.basename(mf)).strip("uvcRP")+ "GSMmodel.abscal.uvcRP") # New path+name of the GSM abscal'd miriad file
    os.system("sh " +apply_abscal+ " " +bp_scaled_calfits+ " " +mf+ " " +new_mf_path)
