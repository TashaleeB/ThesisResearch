from astropy.io import fits
import numpy as np, matplotlib.pyplot as plt
import sys, glob
from matplotlib import gridspec

#%%%
norm = True

#-----------------------------------------------------------------------------
# This section provides a quick way to view calibrated Stokes Parameters.
#-----------------------------------------------------------------------------

fitslist = glob.glob('*.fits') #search for 4 calibrated/polarized fits images
#npzlist = glob.glob('*.npz')

hdulist = fits.open('4pol2457548.45923.fits')
hdr = hdulist[0].header

racen,deccen = hdr['CRVAL1'],hdr['CRVAL2']
racenpix,deccenpix = int(hdr['CRPIX1'])-1, int(hdr['CRPIX2'])-1
radelt,decdelt = hdr['CDELT1'],hdr['CDELT2']

ramin = racen + (radelt*racenpix)
ramax = racen - (radelt*racenpix)
decmin = deccen - (decdelt*deccenpix)
decmax = deccen + (decdelt*deccenpix)
extent = (ramin,ramax,decmin,decmax)

nrow,ncol = 2,2

top=1.-0.5/(nrow+1)
bottom=0.5/(nrow+1)
left=0.5/(ncol+1)
right=1-0.5/(ncol+1)

f,axarr = plt.subplots(nrow,ncol,sharex=True,sharey=True)
gs = gridspec.GridSpec(nrow,ncol, width_ratios=[1, 1], wspace=0.01, hspace=0.1, top=top,
                       bottom=bottom, left=left, right=right) 
iarr = [0,0,1,1]
jarr = [0,1,0,1]

for i in range(4):
     print(i)
     ax = plt.subplot(gs[iarr[i],jarr[i]])
     if not norm:
         if i == 0:
             vmax = 0.6
             vmin = -0.025
             cmap = 'viridis'
             print('one')
         else:
             vmax = 0.008
             vmin = -0.008
             cmap = 'RdYlGn'
             print('two')
     else:
         if i==0:
             vmax = 1.
             vmin = -0.04
             cmap = 'viridis'
             print('three')
         else:
             vmax = 0.01
             vmin = -0.01
             cmap = 'RdYlGn'
             print('four')
     if i<=1:
         ax.set_xticklabels([])
         print('five')
     else:
         ax.set_xlabel('R.A. [deg.]',size=12)
         print('six')
     if i in [1,3]:
         #ax.set_yticklabels([])
         print('yay')
         print('seven')
     else:
         ax.set_ylabel('Dec. [deg.]',size=12)
         print('eight')
     if norm:
         N = 0.6
         print('nine')
     else:
         N = 1.
     img = hdulist[0].data[i,0,:,:][::-1]/N #np.arcsinh(hdulist[0].data[i,0,:,:])[::-1]
     im = ax.imshow(img,cmap=cmap,vmax=vmax,vmin=vmin,extent=extent)
     f.colorbar(im,ax=ax)
     print('ten')
#plt.savefig('./image_7548.pdf')
plt.suptitle('Stokes Plots')
plt.show()

f = plt.figure(figsize=(10,7))
for pol in np.arange(4):
    fig = aplpy.FITSFigure(fitsfile,dimensions=[0,1],slices=[0,pol],figure=f,subplot=(2,2,pol+1))
    if pol == 0:
        vmax=5
        vmin=-0.5
    else:
        vmax = 1
        vmin = -1
    fig.show_colorscale(cmap='jet',vmax=vmax,vmin=vmin)#,stretch='arcsinh')
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
#%%

#-----------------------------------------------------------------------------
# This section allows you the following
# - how the Stokes Paramters vary from the average
# - how gain solution vs freq vary from the average
#-----------------------------------------------------------------------------
nrow,ncol = 2,4
iarr = [0,0,1,1,0,0,1,1]
jarr = [0,1,0,1,2,3,2,3]

top=1.-0.5/(nrow+1)
bottom=0.5/(nrow+1)
left=0.05
right=0.99

goodantenna = [9,10,20,31,53,64,65,72,88,89,96,97,104,105,112] # Antenna with absolute calibrations
fitslist = ['clean2_2457548.R.fits','clean2_2457549.R.fits','clean2_2457550.R.fits','clean2_2457551.R.fits',
            'clean2_2457552.R.fits','clean2_2457553.R.fits','clean2_2457554.R.fits','clean2_2457555.R.fits']
npzlist = ['2457548.R.npz','2457549.R.npz','2457550.R.npz','2457551.R.npz',
           '2457552.R.npz','2457553.R.npz','2457554.R.npz','2457555.R.npz']
hdulist = fits.open(fitslist[0])

shape=(len(fitslist),(hdulist[0].data).shape[0],(hdulist[0].data).shape[2],(hdulist[0].data).shape[3])
#    (num of days  ,num of stokes params      ,???                       ,???                       )
arrayoffitdata = np.ndarray(shape)
arrayofnpzdata = np.ndarray((8,2, 1024, 113), dtype=complex)

for fit in fitslist: #build array of data
    daynum = np.where(np.array(fitslist)==fit)[0][0]
    file = fits.open(fit)
    hdu = file[0].data[:,0,:,:][::-1]
#    hdu = file[0].data[:,0,:,:][::-1]/N
    arrayoffitdata[daynum] = hdu
for npz in npzlist: #build array of data
    daynum = np.where(np.array(npzlist)==npz)[0][0]
    file = np.load(npz)
    gain = file['gains']
    arrayofnpzdata[daynum] = gain

meanI = np.mean(arrayoffitdata[:,0,:,:],axis=0) # 512x512 array for average of stokes I
npzmeanx = np.mean(arrayofnpzdata[:,0,:,:], axis=0) # 1024x113 array for x complex gain solutions
npzmeany = np.mean(arrayofnpzdata[:,1,:,:],axis=0) # 1024x113 array for y complex gain solutions

f2,axarr2 = plt.subplots(nrow,ncol,sharex=True,sharey=True,figsize=(14,8))
gs = gridspec.GridSpec(nrow,ncol, wspace=0.2, hspace=0.2, top=top,
                       bottom=bottom, left=left, right=right)
day=0
numplots = nrow*ncol
stokparam = 3 #[I,Q,U,V] = [3,2,1,0]
sp = ['V','U','Q','I'][stokparam]

for ind in range(numplots):
    ax=plt.subplot(gs[iarr[ind],jarr[ind]])
    if ind == 0:
        yfunc = arrayoffitdata[day,stokparam,:,:][::-1]
        ax.set_ylabel(sp+' Dec. [deg.]',size=12)
        cmap='magma'
    if ind == 1:
        yfunc = (arrayoffitdata[day,stokparam,:,:] - np.mean(arrayoffitdata[:,stokparam,:,:],axis=0)
                 )[::-1]
        #ax.set_ylabel('Deviation From Mean ('+sp+')',size=12)
        cmap='seismic'
    if ind == 2:
        yfunc = np.var(arrayoffitdata[:,stokparam,:,:],axis=0,ddof=0)[::-1]
        ax.set_ylabel('Spread of '+sp+' Over 8 Days',size=12)
        cmap='magma'
    if ind == 3:
        yfunc = np.std(arrayoffitdata[:,stokparam,:,:],axis=0,ddof=0)[::-1]
        #ax.set_ylabel('Standard Deviation ('+sp+')',size=12)
        cmap='magma'
    if ind == 4:
        yfunc = np.abs(arrayofnpzdata[day,0,:,:])[:,goodantenna].T #x gains --> [day,1,:,:] for y gain
        #ax.set_ylabel('X Gain Solutions',size=12)
        cmap='magma'
    if ind == 5:
        yfunc = np.abs(npzmeanx - arrayofnpzdata[day,0,:,:])[:,goodantenna].T
        ax.set_ylabel('Deviation From Mean',size=12)
        cmap='magma'
    if ind == 6:
        yfunc = np.abs(np.var(arrayofnpzdata[:,0,:,:],axis=0,ddof=0,dtype=complex))[:,goodantenna].T
        #ax.set_ylabel('Spread of X-Gain Over 8 Days',size=12)
        cmap='magma'
    if ind == 7:
        yfunc = np.abs(np.std(arrayofnpzdata[:,0,:,:],axis=0,ddof=0,dtype=complex))[:,goodantenna].T
        ax.set_ylabel('Standard Deviation (X-Gain)',size=12)
        cmap='magma'
    if ind<= ncol-1:
        #ax.set_xticklabels([])
        ax.set_xlabel('R.A. [deg.]',size=12)
    else:
        ax.set_xlabel('Frequency [chan]',size=12)
    im = ax.imshow(yfunc,cmap=cmap,aspect='auto', interpolation='nearest',alpha=1.0)
    f2.colorbar(im,ax=ax)
plt.suptitle('Day '+str(day+1))
plt.savefig('Day'+str(day+1)+'Stokes'+sp+'.pdf')
#plt.show()

