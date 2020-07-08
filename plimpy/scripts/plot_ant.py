"""
    Plot the antennas in a pyuvdata object
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from pyuvdata import UVData

filename = ''

# read in pyuvdata file
uv = UVData()
uv.read(filename)

# get antenna positions and the names associated
antpos_xyz, antnum = uv.get_ENU_antpos()

# Plot antennas with names
plt.plot(antpos_xyz[:,0],antpos_xyz[:,1],'o')
for iant,ant in enumerate(antnum):
    plt.text(antpos_xyz[iant,0],antpos_xyz[iant,1],str(ant))

# Plot uv coverage
plt.plot(uv.uvw_array[:,0],uv.uvw_array[:,1],'.')
plt.plot(uv.uvw_array[:,0],-uv.uvw_array[:,1],'.')
plt.axis('equal')

# select good antennas and create a new pyuvdata object
uvgood = uv.select(antenna_nums=[0,1,13,23,25,26], inplace=False)
