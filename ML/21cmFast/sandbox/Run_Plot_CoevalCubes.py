# Following a tutorial found:
# https://21cmfast.readthedocs.io/en/latest/tutorials/coeval_cubes.html

import matplotlib.pyplot as plt
import os
# We change the default level of the logger so that
# we can see what's happening with caching.
import logging, sys, os
logger = logging.getLogger('21cmFAST')
logger.setLevel(logging.INFO)

import py21cmfast as p21c

# For plotting the cubes, we use the plotting submodule:
from py21cmfast import plotting

# For interacting with the cache to clear the cache so that we get the same results for the notebook every time.
from py21cmfast import cache_tools

#Clear the Cache
if not os.path.exists('_cache'):
    os.mkdir('_cache')

p21c.config['direc'] = '_cache'
cache_tools.clear_cache(direc="_cache")

# The simplest and most efficient way to produce a coeval cube is simply to use the "run_coeval"
# The output of run_coeval is a list of Coeval instances, one for each input redshift.
coeval8, coeval9, coeval10 = p21c.run_coeval(
    redshift = [8.0, 9.0, 10.0], #array_like -> A single redshift or multiple redshift.
    user_params = {"HII_DIM": 100, "BOX_LEN": 100, "USE_INTERPOLATION_TABLES": True}, # 100 Mpc box len
    cosmo_params = p21c.CosmoParams(SIGMA_8=0.8), #cosmological parameters used to compute initial conditions.
    astro_params = p21c.AstroParams({"HII_EFF_FACTOR":20.0}), #astrophysical parameters defining the course of reionization.
    random_seed=12345
)

# By default, each of the components of the cube are cached to disk (in our _cache/ folder) as we run it. However, the Coeval cube itself is not written to disk by default.
filename = coeval8.save(direc='_cache')
coeval8.init_struct.save(fname='my_init_struct.h5')

# You can read in the Coeval cube
new_coeval8 = p21c.Coeval.read(filename, direc='.')

# prints the shape of the fields
print(coeval8.hires_density.shape)
print(coeval8.brightness_temp.shape)

# List the kind of field
 coeval8.get_fields()
 coeval8.cosmo_params.cosmo

# Using the plotting function
fig, ax = plt.subplots(1,3, figsize=(14,4))
for i, (coeval, redshift) in enumerate(zip([coeval8, coeval9, coeval10], [8,9,10])):
    plotting.coeval_sliceplot(coeval, kind='brightness_temp', ax=ax[i], fig=fig);
    plt.title("z = %s"%redshift)
plt.tight_layout()
plt.show

fig, ax = plt.subplots(1,3, figsize=(14,4))
for i, (coeval, redshift) in enumerate(zip([coeval8, coeval9, coeval10], [8,9,10])):
    plotting.coeval_sliceplot(coeval, kind='density', ax=ax[i], fig=fig);
    plt.title("z = %s"%redshift)
plt.tight_layout()
plt.show
