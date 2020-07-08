from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import PIL
import time, sys,argparse

from glob import glob
from astropy.io import fits
from pixell import enmap, utils
from pixell import reproject
from pixell import enplot
from IPython.display import clear_output
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')
rc('font',size=16)

"""
    This was based on https://nbviewer.jupyter.org/github/UPennEoR/plimpy/blob/master/PolarizedMosaics/HERAPixellMosaicWithMueller.ipynb
"""

def progress_bar(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))
    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)

# Convenient wrapper for displaying enmaps
def eshow(x,**kwargs): enplot.show(enplot.plot(x,**kwargs))

# Define Arguments
a = argparse.ArgumentParser(description="Run using python as: python -c commands_PolarizedMosaic.py <args>")
a.add_argument('--file_path', '-fp', type=str, help='Path + the name of text file without extension',
               required=True)

if __name__ == '__main__':
    
    args = a.parse_args()
    
    path = args.file_path #'/lustre/aoc/projects/hera/tbilling/'
    fitsfiles = sorted(glob(path+'*.image.fits'))
    
    # Define some dimensions
    fitsfile = fitsfiles[0]
    nmaps = len(fitsfiles)
    
    # Read the map
    imap = enmap.read_map(fitsfile,) # (4, 1, 512, 512)
    # For some reason, pixell makes it really hard to get at the actual values in the WCS header
    hdu = fits.open(fitsfile)
    # Define the size of the pixels in the uber-map as a fraction of the size of the original map
    pix_fac = 2.
    pixsize_new = np.abs(imap.wcs.pixel_scale_matrix[0][0])*60./pix_fac * utils.arcmin

    # Need to trim mastermap ... a lot
    # Figure out the extents
    for ifile,fitsfile in enumerate(fitsfiles):
        print(ifile+1)
        progress_bar((ifile+1)/nmaps)
        hdu = fits.open(fitsfile)
        # Extract a postage stamp centered on the center of the image
        dec0, ra0 = np.deg2rad(np.array((hdu[0].header['CRVAL2'],hdu[0].header['CRVAL1'])))
        width = np.deg2rad(20.)
        # [[decfrom,rafrom],[[decto,rato]]
        this_decfrom = dec0-width/2.
        this_rafrom = ra0-width/2.
        this_decto = dec0+width/2.
        this_rato = ra0+width/2.
        if ifile == 0:
            decfrom = this_decfrom
            rafrom = this_rafrom
            decto = this_decto
            rato = this_rato
        else:
            if this_decfrom < decfrom:
                decfrom = this_decfrom
            if this_rafrom < rafrom:
                rafrom = this_rafrom
            if this_decto > decto:
                decto = this_decto
            if this_rato > rato:
                rato = this_rato
    
    # Define the master map and define its shape
    shape_band, wcs_band = enmap.band_geometry(np.radians([-60,-10]),
                                               res=pixsize_new, proj='car') #shape=shape_fullsky, proj='car')
    mastermap = enmap.enmap(np.zeros([4,1,shape_band[0],shape_band[1]]), wcs=wcs_band).submap([[decfrom,rato],[decto,rafrom]])
    shape_band = mastermap.shape # (4, 1, 291, 2420)
    wcs_band = mastermap.wcs # car:{cdelt:[-0.06944,0.06944],crval:[0,0],crpix:[2329,589]}

    # Note the reversed order
    nstokes, nfreq, ny, nx = shape_band

    # Note that pixell wants last two trailing dimensions to be the ones associated with the WCS;
    # the others are arbitrary
    pStokes = enmap.enmap(np.zeros([4,nfreq,nmaps,ny,nx]), wcs=wcs_band) # (4, 1, 56, 291, 2420)
    matrix = np.zeros([len(fitsfiles),pStokes.shape[-2],pStokes.shape[-1]])

    for ifile,fitsfile in enumerate(fitsfiles):
        progress_bar((ifile+1)/nmaps)
        # Read file into pixell object
        imap = enmap.read_map(fitsfile,) # (4, 1, 512, 512)
        # Extract a postage stamp centered on the center of the image
        hdu = fits.open(fitsfile)
        dec0, ra0 = np.deg2rad(np.array((hdu[0].header['CRVAL2'],hdu[0].header['CRVAL1'])))
        width = np.deg2rad(20.)
        box = np.array([[dec0-width/2.,ra0-width/2.],[dec0+width/2.,ra0+width/2.]])
        
        stamp = imap.submap(box)
        
        # project the stamp into the master grid and turn all zeros to np.nan
        stampmap = enmap.project(stamp, shape_band, wcs_band)
        t0 = time.time()
        
        # Loop over each pixel per stokes parameter per frequency
        for s in range(nstokes):
            for f in range(nfreq):
                for y in range(stampmap.shape[-2]):
                    for x in range(stampmap.shape[-1]):
                        #replace all zeros with np.nan
                        if stampmap[s,f,y,x] == 0:
                            stampmap[s,f,y,x] = np.nan
        t1 = time.time()
        print('Elapsed time',t1-t0,'seconds')
    
        # Store all the stamps
        pStokes[:,:,ifile,:,:] = stampmap # (4, 1, 56, 291, 2420)
        matrix[ifile,:,:] = stampmap # (56, 291, 2420)
        
        # Accumulate
        #mastermap += stampmap # (4, 1, 291, 2420)

    # Write the output to a fits file.
    enmap.write_fits(path+'matrixmaps.fits', matrix)
    
    #Combine maps by taking the average of each pixel over all of the images
    meany, meanx = pStokes.shape[-2], pStokes.shape[-1] #ycoord and xcoord
    mastermap = np.zeros([len(fitsfiles),pStokes.shape[-2],pStokes.shape[-1]])
    for s in range(nstokes):
        for f in range(nfreq):
            for my in range(meany):
                progress_bar(my/meany)
                for mx in range(meanx):
                    #create array that will be used to caluclate the mean at a pixel
                    pixel_array = np.array([])
                    for i in range(nmaps):
                        #select a pixel and add it to 1D array containing nmap pixels
                        pixel = matrix[i,my,mx]
                        pixel_array = np.append(pixel_array,pixel)
                        # Once the 1D array is "full", take the nanmean and append it to mastermap
                        if pixel_array.shape[0] == nmaps:
                            mean_pixel = np.nanmean(pixel_array)
                            mastermap[s,f,y,x] = mean_pixel
    
    # write the output back to fits
    enmap.write_fits(path+'mastermap.fits', mastermap)
    enmap.write_fits(path+'pStokes.fits', pStokes)
    #hdu = fits.PrimaryHDU(mastermap)
    #hdu.writeto('mastermap.fits',overwrite=True)
    #hdu = fits.open('I.fits')
    #data = hdu[0].data

    # Build Mueller
    muellermodel = '/lustre/aoc/projects/hera/gtucker/notebooks/muellerbeam.fits'
    hdu = fits.open(muellermodel)
    mueller_data = hdu[0].data # (4, 4, 1, 512, 512)
    # Define the pixell map stack ...
    Mueller = enmap.enmap(np.zeros([4,4,nfreq,nmaps,ny,nx]), wcs=wcs_band)

    # The trick here is to use the wcs information from each file you read in to define the Mueller matrix
    for ifile,fitsfile in enumerate(fitsfiles):
        progress_bar(ifile/nmaps)
        # Read file into pixell object
        imap = enmap.read_map(fitsfile,)
        # Extract a postage stamp centered on the center of the image
        hdu = fits.open(fitsfile)
        dec0, ra0 = np.deg2rad(np.array((hdu[0].header['CRVAL2'],hdu[0].header['CRVAL1'])))
        width = np.deg2rad(20.)
        box = np.array([[dec0-width/2.,ra0-width/2.],[dec0+width/2.,ra0+width/2.]])
        
        this_mueller = enmap.enmap(mueller_data, wcs = imap.wcs)
        
        mueller_stamp = this_mueller.submap(box)
        
        # project the stamp into the master grid
        mueller_stampmap = enmap.project(mueller_stamp, shape_band, wcs_band)
        
        # Store all the stamps
        Mueller[:,:,:,ifile,:,:] = mueller_stampmap

    # write the output back to fits
    enmap.write_fits(path+'Mueller.fits', Mueller)

    ndata = 4*nmaps
    # We will probably always assume W is diagonal; but there might be noise correlation between the pseudo-Stokes ... ?
    W = np.eye(ndata) # (224, 224)

    # There's probably a faster way, but the brute force approach is to construct the linear combination at every pixel

    failed, npix = 0, 0
    xrange = np.arange(nx)
    yrange = np.arange(ny)
    t0 = time.time()
    Stokes = enmap.enmap(np.zeros(mastermap.shape), wcs = mastermap.wcs)

    for inx in xrange:
        progress_bar(npix/(len(xrange)*len(yrange)))
        for iny in yrange:
            npix += 1
            # Construct A and the data
            A = np.zeros([ndata,4])
            d = np.zeros([ndata])
            for imap in np.arange(nmaps):
                indx = np.arange(imap*4, (imap+1)*4)
                A[indx, :] = Mueller[:, :, 0, imap, iny, inx]
                d[indx] = pStokes[:, 0, imap, iny, inx]
            ATWd = np.matmul(A.T, d)
            Cov = np.matmul(A.T, A)
            # Need to think about computing the condition number and the inversion method here
            try:
                invCov = np.linalg.inv(Cov)
                bestfit = np.matmul(invCov, ATWd) # (1, 4)
            except:
                bestfit = np.array([np.nan, np.nan, np.nan, np.nan])
                failed += 1
            Stokes[:,0,iny,inx] = bestfit
    t1 = time.time()
    print('Elapsed time',t1-t0,'seconds')
