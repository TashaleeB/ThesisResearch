import os
from pyuvdata import UVData

# Convert Amp Scale to hdf5
npz2h5 = ("python absref_npz_to_hdf5.py --fname=SpectrumScale_simOVERreal.npz  --outname=SpectrumScale_simOVERreal.h5 --npzkey=SpectrumScale --h5key=spectrum_scale")

# Convert CASA calibration solutions to hdf5
cal2h5 = ("casa --agg -c casa_cal_to_hdf5.py --Kcal=gc.2457548.uvcRP.abscaltest.MS.2K.cal --Bcal=gc.2457548.uvcRP.abscaltest.MS.2B.cal")


# Combine hdf5 and convert to calfits file
h52calfits = ("python hdf5_to_calfits.py --fname=bandpassWdelay_scale.calfits --uv_file=gc.2457548.uvcRP --bcal=../gc.2457548.uvcRP.forceQUV2zero.MS.B.cal.h5 --acal=../SpectrumScale_forceQUV2OVERsim.h5 --overwrite --multiply_gains --smooth_ratio --plot_ratio")

#--kcal=gc.2457548.uvcRP.abscaltest.MS.2K.cal.h5

# Apply calfits file to miriad file and save it as a new miriad file
applycal = ("sh apply_abscal.sh bandpassWdelay_scale.calfits gc.2457548.uvcRP gc.2457548.uvcRP.abscal.uvcRP")

os.system(npz2h5)
os.system(cal2h5)
os.system(h52calfits)
os.system(applycal)

# Convert Miriad file to uvfits
uv = UVData()
uv.read_miriad("gc.2457548.uvcRP.abscal.uvcRP")
uv.write_uvfits('gc.2457548.uvcRP.abscal.uvcRP.uvfits',spoof_nonessential=True,force_phase=True)
