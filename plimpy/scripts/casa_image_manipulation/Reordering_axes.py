#!/bin/sh

#  Reordering_axes.py
#  
#
#  Created by Tashalee Billings on 12/18/18.
#  

"""
    This script is used to reorder the axes in a CASA image file that was generated from a FITS file.
"""

import argparse

args = argparse.ArgumentParser(description="When you export the FITS image to CASA formatted image the Spectrum and Stokes axes needs to be reordered.")

args.add_argument("imput_image", type=str, nargs='*', help='Input CASA image file that needs axes reordered.')
args.add_argument("output_image", type=str, nargs='*', help='Output CASA image file that needs axes reordered.')

if __name__ == "__main__":
    
    arg = args.parse_args()
    
    # Check to make sure the image file is in the incorrect order,
    # (['Right Ascension', 'Declination', 'Frequency', 'Stokes'])
    ia.open(arg.imput_image)
    s = ia.summary(list=False)       # store header in record
    
    if s['axisnames'][2] == 'Stokes' or s['axisnames'][3] == 'Frequency' :
        raise ValueError("Your CASA image file has the following axis name: ['Right Ascension', 'Declination', 'Stokes', 'Frequency']. No need to perform correction.")
    else:
        print("Axis name: ['Right Ascension', 'Declination', 'Frequency', 'Stokes']. Changing the ordering 'Frequency' and 'Stokes' axis now...")
    ia.close()

    # Reorder Axis
    imtrans(imagename=arg.imput_image, outfile=arg.output_image, order="0132")
