# look at visibilities in pyuvdata

from pyuvdata import UVData

import matplotlib.pyplot as plt
import numpy as np

filename = "zen.2458560.69133.HH.uvh5"
uv = UVData()
uv.read(filename)

ants = np.unique(uv.ant_1_array)

plt.figure(figsize=(12,8))
for ant in ants:
    uv.select(ant_str='{}x_{}x'.format(ant,ant))
    data = (uv.data_array).squeeze()
    plt.plot(data.T.real)
    
    del uv
    uv = UVData()
    uv.read(filename)

plt.savefig("visibility_{}.png".format(filename))
plt.show()



>>> from pyuvdata import UVData
>>> import numpy as np
>>> UV = UVData()
>>> filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
>>> UV.read(filename)
>>> data = UV.get_data(1, 2, 'rr')  # data for ant1=1, ant2=2, pol='rr'
>>> times = UV.get_times(1, 2)  # times corresponding to 0th axis in data
>>> print(data.shape)
(9, 64)
>>> print(times.shape)
(9,)

# One can equivalently make any of these calls with the input wrapped in a tuple.
>>> data = UV.get_data((1, 2, 'rr'))
>>> times = UV.get_times((1, 2))
