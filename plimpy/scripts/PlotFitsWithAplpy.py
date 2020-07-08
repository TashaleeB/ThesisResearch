#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:36:53 2018

@author: jaguirre
"""
#printf "\e[?2004l"

#You can use this instead of DS9

import numpy as np
import matplotlib.pyplot as plt
import aplpy
from astropy import units as u
from astropy.io import fits
#%matplotlib notebook

#fitsfile='zen.2458050.50580.HHPMNFor.MS.phs.image.fits'
fitsfile = '2458115.24482.uvOCR.fits'

f = plt.figure(figsize=(10,7))
for pol in np.arange(4):
    fig = aplpy.FITSFigure(fitsfile,dimensions=[0,1],slices=[0,pol],figure=f,subplot=(2,2,pol+1))
    if pol == 0:
        vmax=18
        vmin=-4
        cmap='viridis'
    else:
        vmax = 1
        vmin = -1
        cmap='RdYlGn'
    fig.show_colorscale(cmap=cmap)#,vmax=vmax,vmin=vmin)#,stretch='arcsinh')
    fig.add_grid()
    fig.grid.set_color('black')
    fig.grid.set_xspacing(15)
    fig.grid.set_yspacing(15)
    fig.grid.show()
    fig.axis_labels.set_font(size='small')
    fig.tick_labels.set_font(size='small')
    fig.tick_labels.set_xformat('hh')
    fig.tick_labels.set_yformat('dd')
    fig.add_colorbar()
    fig.colorbar.set_font(size='small')
    
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from glob import glob
from os import system
import aplpy
from astropy import units as u
from astropy.io import fits
#%matplotlib notebook

#fitsfile='zen.2458050.50580.HHPMNFor.MS.phs.image.fits'
fitsfile = '2458115.24482.uvOCR.fits'

f = plt.figure(figsize=(10,7))
for pol in np.arange(4):
    fig = aplpy.FITSFigure(fitsfile,dimensions=[0,1],slices=[0,pol],figure=f,subplot=(2,2,pol+1))
    if pol == 0:
        vmax=1.7
        vmin=-0.2
        cmap="viridis"
    if pol == 1:
        vmax=0.3
        vmin=-.2
        cmap="PRGn"
    if pol == 2:
        vmax=0.1
        vmin=-0.03
        cmap="PRGn"
    if pol == 3:
        vmax=0.003
        vmin=-0.007
        cmap="PRGn"

    fig.show_colorscale(cmap='jet',vmax=vmax,vmin=vmin)#,stretch='arcsczdxcinh')
    fig.add_grid()
    fig.grid.set_color('black')
    fig.grid.set_xspacing(15)
    fig.grid.set_yspacing(15)
    fig.grid.show()
    fig.axis_labels.set_font(size='small')
    fig.tick_labels.set_font(size='small')
    fig.tick_labels.set_xformat('hh')
    fig.tick_labels.set_yformat('dd')
    fig.add_colorbar()
    fig.colorbar.set_font(size='small')

fig.savefig('{}.png'.format(fitsfile))
plt.show()
