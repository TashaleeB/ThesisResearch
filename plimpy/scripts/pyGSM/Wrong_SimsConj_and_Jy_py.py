from pyuvdata import UVData

prefix = args.prefix
suffix = args.suffix

files = []
for p in pols:
    files.append(prefix+'.'+p+suffix)

uv = UVData()
print 'Reading data'
for f in files:
    print f
uv.read_miriad(files)

# Does Conj and proper Jy Units
k_b = 1.38064852e-23
lamb = 2 #[meter]
jansky_conver = 1e26*2*k_b/np.power(lamb,2) #[Jy/sr*1/K]
uv.data_array = np.conj(uv.data_array)*jansky_conver #for Sim Data

miriadfile = prefix+'.conj.Jy'+suffix
print 'Writing data to '+miriadfile
uv.write_miriad(miriadfile)