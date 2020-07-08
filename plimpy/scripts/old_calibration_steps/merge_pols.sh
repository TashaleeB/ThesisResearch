#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:17:23 2018

@author: jaguirre

chmod u+x merge_pol.sh
./merge_pols.sh zen.2458050.50580
"""

#!/bin/bash
PREFIX=$1
echo 'Using prefix '$PREFIX
python merge_pols_uv.py $PREFIX
echo 'Converting to uvfits'
# Can we change default output name?
miriad_to_uvfits.py $PREFIX'.HH.uv'