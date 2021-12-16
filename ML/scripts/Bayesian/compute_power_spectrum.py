"""
https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/
https://stats.stackexchange.com/questions/211436/why-normalize-images-by-subtracting-datasets-image-mean-instead-of-the-current
http://www.rctn.org/bruno/npb261b/lab2/lab2.html

A power spectrum is an analysis tool that is very often used to get some statistical information.

1. Convert the data set into a suitable data array with the correct spatial layout.
2. Take the Fourier transform of the data array in the number of dimensions of the data array.
3. Construct a corresponding array of wave vectors, k, with the same layout as the Fourier amplitudes.
4. Bin the amplitudes of the Fourier signal into, k, bins and compute the total variance within each bin.

"""

import skimage, scipy.stats as stats, numpy as np, matplotlib.pyplot as plt
import matplotlib.image as mpimg # allows you to import images back into arrays

from scipy.stats import binned_statistic
from matplotlib import gridspec
from matplotlib.ticker import PercentFormatter

wedge = False # Is the data wedge filtered
data_path = "/ocean/projects/ast180004p/tbilling/data/"

if wedge == False:
    inputFile = data_path+"t21_snapshots_nowedge_v12.hdf5"
    perfectmodel = "/ocean/projects/ast180004p/tbilling/sandbox/redo_mlpaper/no_modes_removed/CNN_model_nowedge_1.h5" # nowedge

if wedge == True:
    inputFile = data_path+"t21_snapshots_wedge_v12.hdf5"
    perfectmodel = "/ocean/projects/ast180004p/tbilling/sandbox/redo_mlpaper/modes_removed/CNN_model_wedge_5.h5" # wedge
    
outputdir = "/ocean/projects/ast180004p/tbilling/sandbox/bayesian/denseflipout/sandbox/"

# ------------
#    STEP 1
# ------------
# load data into memory
image = np.load()[0,:,:,0]

# visualize imported image
#plt.imshow(image, cmap='gray')
#plt.show()

# The analysis will only work for a square image
if image.shape[0] == image.shape[1] :
    # The pixel resolution of the image will be important for the remainder of the analysis
    npix = image.shape[0]
    
# ------------
#    STEP 2
# ------------

# To take the Fourier transform of our two dimensional image data array, we will use numpy.fft.fftn, multi dimensional function, here as that makes it possible to generalise the technique to three dimensional
fourier_image = np.fft.fftn(image)

# The Fourier image array now contains the complex valued amplitudes of all the Fourier components. We are only interested in the size of these amplitudes. We will further assume that the average amplitude is zero, so that we only require the square of the amplitudes to compute the variances.
fourier_amplitudes = np.abs(fourier_image)**2

# ------------
#    STEP 3
# ------------
# To bin the results found above in, k-space, we need to know what the layout of the return value of numpy.fft.fftn is: what is the wave vector corresponding to an element with indices i and j in the return array?
L =  2000 #[Mpc/h]
kfreq = np.fft.fftfreq(npix, d=L/npix) * 2* np.pi # return a one dimensional array containing the wave vectors

# To convert this to a two dimensional array matching the layout of the two dimensional Fourier image, we can use numpy.meshgrid
kfreq2D = np.meshgrid(kfreq, kfreq)

# convert wave vector into norm
knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

# flatten wave vector norms and Fourier image
knrm = knrm.flatten()
fourier_amplitudes = fourier_amplitudes.flatten()

# ------------
#    STEP 4
# ------------
# To bin the amplitudes in k-space, we need to set up wave number bins by creating integer k-value bins
kbins = np.arange(0.5, npix//2+1, 1.) # the maximum wave number will equal half the pixel size of the image

# The kbin array will contain the start and end points of all bins; the corresponding k-values are the midpoints of these bins:
kvals = 0.5 * (kbins[1:] + kbins[:-1])

"""
Data binning, bucketing is a data pre-processing method used to minimize the effects of small observation errors. The original data values are divided into small intervals known as bins and then they are replaced by a general value calculated for that bin. This has a smoothing effect on the input data and may also reduce the chances of overfitting in the case of small datasets
There are 2 methods of dividing data into bins:

Equal Frequency Binning: bins have an equal frequency.
Equal Width Binning : bins have equal width with a range of each bin are defined as [min + w], [min + 2w] …. [min + nw] where w = (max – min) / (no of bins).
"""
# use scipy.stats to compute the average Fourier amplitude (squared) in each bin
Abins, bin_edges, binnumber = stats.binned_statistic(knrm, fourier_amplitudes,
                                     statistic = "mean",
                                     #bins = kbins)
                                     bins = len(kbins))
# To get the total power within each bin, we need to multiply the average power with the volume in each bin (in 2D, this volume is actually a surface area)
Abins[1:] *= 2.* np.pi * (kbins[1:]**2 - kbins[:-1]**2)

# plot the resulting power spectrum as a function of wave number, P(k) (typically plotted on a double logarithmic scale)
plt.loglog(kvals, Abins[1:])
plt.xlabel("$k$")
plt.ylabel("$P(k)$")
plt.tight_layout()
plt.savefig("power_spectrum.png", dpi = 300, bbox_inches = "tight")
