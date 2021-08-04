import numpy as np, matplotlib.pyplot as plt, scipy.ndimage as spimg, time

# Setup
npix = 32
img_size = np.array([npix, npix])
var_img = 1.
sigma_filt = 2

n_img = 10000
training_data = np.zeros([npix, npix, n_img])
labels = np.zeros(n_img)

def StdNorm(array):
    
    tmp = array.copy()
    std = tmp.std()
    mn = tmp.mean()
    
    return (array - mn)/std

def UnitNorm(array):
    
    array_min = array.min()
    array_max = array.max()
    
    return (array - array_min)/(array_max - array_min)

class Timer():
    
    def start(self):
        self.t0 = time.time()
        #print(self.t0)
        
    def stop(self, message):
        self.t1 = time.time()
        print(message, self.t1 - self.t0, 'sec')

timer = Timer()
timer.start()
for n in range(n_img):
    #print(n)
    img = StdNorm(spimg.gaussian_filter(np.random.normal(loc=0, scale=var_img, size=img_size), sigma_filt))
    
    # Edge cases
    thresh_good = False
    while not thresh_good:
        # A Gaussian gives a more uniform distribution over ionization fractions 
        # (equally likely to get any ionization fraction)
        thresh = np.random.normal(0,1) #(-2,2,1) 
        img_thresh = img.copy()
        ionized = img_thresh > thresh
        frac = ionized.sum()/(img_size.prod())
        #print(frac)
        if frac < 0.995 and frac > 0.005:
            #print("What the fuck?")
            thresh_good = True
        
    img_thresh[img_thresh > thresh] = img_thresh.min()
    img = UnitNorm(img_thresh)
    if np.isfinite(img).sum() != img_size.prod():
        print('Still no good!')
        break
    
    training_data[:,:,n] = img
    labels[n] = frac

timer.stop(str(n_img)+' data points took')

# Make image and histogram plots
"""
fig, ax = plt.subplots(nrows = 3, ncols = 3, figsize=(10,10))
for i in range(3):
    for j in range(3):
                
        im = ax[i][j].imshow(training_data[:,:,i+3*j])
        ax[i][j].set_title('{:.3f}'.format(labels[i+3*j]))
        fig.colorbar(im, ax=ax[i][j], shrink=0.8)
        #ax[i][j].colorbar()

plt.hist(labels, bins=30)
"""
# Save Toy model
np.savez('toy_models.npz', training_data = training_data, labels = labels)
