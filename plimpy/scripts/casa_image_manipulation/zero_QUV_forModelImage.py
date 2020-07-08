#!/bin/sh

#  zero_QUV_forModelImage.py
#  
#
#  Created by Tashalee Billings on 10/25/18.
#  
"""
    Force Stokes Model column QUV to be zero.
"""

import numpy as np

model_name = "" # This is only true for .model files. The variable column may vary from image file to image file.

tb.open(model_name,nomodify=False) # It's important to set the modify param to false when you open the file because this allows you to make changes to the data.
var = tb.getvarcol('map') # I'm making a copy of the entire dataset preserving the dimensions (pix,pix,stokes,freq,spw??)

#Find the pixel values QUV that are not zero and then set them to zero.

for i in range(1,4): #Stokes QUV
    pixels1,pixels2 = np.where(var['r1'][:,:,i,:,:] !=0)
    length = pixels1.shape[0]
    for l in range(length):
        var['r1'][pixels1[l],pixels2[l],i,:,:] *=0

tb.putvarcol('map',var)
tb.close()v
"""
-dirty image
- spectrum of the real data
- Sim/real spectrum
- gain plot
"""
