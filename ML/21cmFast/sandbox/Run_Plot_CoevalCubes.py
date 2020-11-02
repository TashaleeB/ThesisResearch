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
