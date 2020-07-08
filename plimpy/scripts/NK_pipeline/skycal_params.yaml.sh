#!/bin/sh

#  skycal_params.yml.sh
#  
#
#  Created by Tashalee Billings on 2/4/19.
#  
# skycal_params.yml
#
# parameter file for casa_imaging/skycal_pipe.py
#
# Notes:
#   All output filenames have an out_dir root (ex. logfile).
#   All input filenames have a work_dir root (ex. gen_model.cl_params.gleamfile)
#   except for data_file which has a data_root root.
# https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html
#---------------------------------------------------------------
# IO Parameters Dictionary
#---------------------------------------------------------------
io :
  work_dir : './'         # directory to work in
  out_dir : './'  # directory to dump all output in
  logfile : 'skycal_out.log'  # logfile
  errfile : 'skycal_err.log' # error file
  joinlog : True       # redirect error output into logfile
  overwrite : True   # overwrite
  verbose : True        # report feedback to standard output

  # path to CASA executable or CASA command
  casa : 'casa'

  # space-delimited casa flags
  casa_flags : '--nologger --nocrashreport --nogui --agg'

  # path to casa_imaging scripts dir
  # if None will try to get it from build
  casa_scripts : None

#---------------------------------------------------------------
# Analysis Parameters Dictionary
#---------------------------------------------------------------
analysis :
  prep_data       : True          # Prepare data for conversion into Measurement Set
  gen_model       : True          # Generate a flux model of the calibration field
  di_cal          : True          # Direction independent calibration
  di_img          : True          # Imaging after DI calibration: MFS and/or spectral cube
  dd_cal          : False         # A constrained direction calibration
  dd_img          : False         # Imaging after DD calibration: MFS and/or spectral cube

#---------------------------------------------------------------
# Observation Parameters Dictionary
#---------------------------------------------------------------
obs :
  # Observation Parameters
  longitude  : 21.4286     # Observer longitude in degrees East
  latitude   : -30.7215    # Observer latitude in degrees North

  # Source name : set to 'drift' if no particular source is desired
  source     : 'Galactic Center'

  # Source coordinates : Set None if no particular source is desired (i.e. drift scan imaging)
  source_ra  : 232.64       # RA in J2000 degrees
  source_dec : -29.0078106     # Dec in J2000 degrees

#---------------------------------------------------------------
# Data Parameters Dictionary
#---------------------------------------------------------------
data :
  # Absolute root path to data files
  data_root : '/lustre/aoc/projects/hera/plaplant/HERA19Golden/RawData/2457548/'

  # Either 1: File path extension from data_root to a glob-parseable
  # data file (miriad, uvfits or MS)
  # Or 2: A file path extension to a CASA measurement set (ending in .ms)
  data_file : 'zen.2457548.46619.*.HH.uvcRP'

#---------------------------------------------------------------
# Algorithm Parameters Dictionary
#---------------------------------------------------------------
algorithm :

  # Get Source Location And Prep Data Parameters
  prep_data :         # Prepare data for conversion into Measurement Set
    duration : 4           # observation duration in minutes
    start_jd  : 2457548     # integer julian date of desired observation day: must match data_file day
    get_filetimes : True    # open files of-interest and to get more precise transit time
    flag_ext    : "O.flags.npz"         # extension to data_file containing RTP flags as an .npz file
    outfile     : "zen.{:.5f}.HH.uvRP"   # Output file template with a JD-format field
    pols :                            # Polarization(s) to search for: .pol. must be part of the filename
      - 'xx'
      - 'yy'
      - 'xy'
      - 'yx'
    antenna_nums : None      # which antenna numbers to read-in and write-out to MS. Default is all.

  # Flux Model Generation Parameters
  gen_model :
    # Component List Parameters
    gleamfile  : "/lustre/aoc/projects/hera/tbilling/polim/GSM_GC_model/swap_GSM_Model_of_GC_Jy_1024StokesIQUV_WithHeaderCopiedToIt.image.fits" # str, path to GLEAM point-source FITS catalogue
    radius     : 20             # float, radius around pointing in degrees to include GLEAM sources
    min_flux   : 0.1            # float, minimum flux cut of gleam sources
    use_peak   : False          # bool, use peak flux or integrated flux from GLEAM
    image      : True           # bool, make a spectral image cube of the model
    freqs      : 100,200,1024   # str, comma-delimited str holding start,stop,Nfreq in MHz. Ex: 100,200,1024
    cell       : 300arcsec      # string, pixel size of image. Ex: 300arcsec
    imsize     : 512            # int, image side-length in pixel units
    # Primary Beam Correction Parameters
    pbcorr       : True           # bool, multiply image by primary beam to get a perceived flux model
    beamfile : "hera_pspec/hera_pspec/data/HERA_NF_dipole_power.beamfits" # str, path to HERA healpix beam model
    pols :                      # Polarization models to use in pb correction
        - 'xx'
        - 'yy'
        - 'xy'
        - 'yx'
    # Time of observation center in Julian Date, corresponding to source_ra
    # Overwritten if prep_data == True and source_ra is not None
    time : None

  # General Calibration Parameters for Both DI and DD Calibration
  gen_cal :
    ex_ants     : 22,43,80,81 # str, comma-delimited str of bad antenna
    export_gains  : True        # export to calfits and optionally smooth. If True, the corresponding miriad visibility file must be present.
    # Gain Export Parameters
    smooth      : True    # smooth with a Gaussian Process
    gp_max_dly  : 200     # Maximum delay [ns] for smoothing
    medfilt     : True    # median filter bandpass before smoothing
    kernel      : 13      # median filter kernel size (channels)
    bp_broad_flags : True    # broadcast flags at a specific channel across all antennas
    bp_flag_frac   : 0.25    # fractiona of flagged antennas needed to broadcast across all antennas

  # Direction Independent Calibration Parameters
  di_cal :
    model : 'gleam.cl.pbcorr.image'    # str, path to flux model. Overwritten by gen_model if True.
    refant : 10           # int or comma-delimited str of reference antenna(s)
    rflag : False         # bool, run rflag task
    KGcal : True          # bool, run Delay and Mean Phase cal
    KGsnr : 0.0           # float, SNR cut in KGcal
    Acal  : True          # bool, run Mean Amp cal
    Asnr  : 0.0           # float, SNR cut in Acal
    BPcal : True          # bool, run Bandpass cal
    BPsnr : 0.0           # float, SNR cut in BPcal
    BPsolnorm : False     # bool, normalize BP amplitude to unity across freq
    uvrange   : ""        # str, UVrange in calibration
    timerange : ""        # str, UTC time-range to use in calibration
    gain_spw  : ""        # str, spw selection for gaincal calls
    bp_spw    : ""        # str, spw selection for bandpass call
    split_cal : False     # bool, split CORRECTED column from MS
    split_ext : "split"   # str, file extension of CORRECTED split
    split_model : True   # str, split MODEL column into a new .model MS
    gain_ext  : ''        # str, file extension of gain tables
    gaintables  : ''      # pre-calibration gain tables to apply first

  # Imaging parameters
  imaging :
    # MFS Imaging Parameters
    image_mfs   : True      # bool, make an MFS image of corrected data with parameters below
    image_mdl   : False     # bool, make an MFS image of the split MODEL
    image_res   : False     # bool, perform a UVsub and make MFS image of residual
    niter       :         # int, total number of CLEAN iterations
    - 100
    - 100
    threshold   :         # str, Flux threshold to stop CLEAN
    - 0Jy
    cycleniter  :         # int, max number of iterations for a minor cycle
    - 1000
    mask        :         # str, CLEAN masks to use
    - clean_reg1.crtf
    - ''
    weighting   : 'briggs'  # str, UV weighting for image
    robust      : -1        # float, robust parameter for briggs weighting
    uvrange     : ''        # str, uvrange for imaging
    pxsize      : 300       # int, pixel size in arcseconds
    imsize      : 512       # int, image side-length in pixel units
    spw         : "0:150~900"        # str, spw selection for MSF image
    stokes      : "IQUV"    # str, Stokes parameters to image
    timerange   : ""        # str, UTC timerange to image
    deconvolver : 'hogbom' # str, deconvolving algorithm to use
    gridder     : 'standard' # str, gridder to use: [standard, wproject]
    wpplanes    : 1         # int, wprojectplanes for wprojection
    pblimit     : -1        # float, pb response cut
    # Spectral Imaging Parameters
    image_spec  : False     # make a coarse spectral cube via multiple MFS images
    image_mdl_spec : False  # make a coarse spec cube of the model
    spec_start  : 0         # int, if image_spec, this is starting channel
    spec_end    : 1024      # int, if image_spec, this is ending channel
    spec_dchan  : 250       # int, if image_spec, this is Nchans per MFS image
    # Source Extraction Parameters
    source_extract  : True    # bool, if image_spec, extract spectrum from output image cube
    pols        : [-5, -6, -7, -8]  # stokes polarizations to extract [xx,yy,xy,yx]
    radius      : 1         # radius around source in degrees to estimate peak flux
    gauss_mult  : 1.5       # Coefficient of PSF width to use as Gaussian fit mask
    plot_fit    : False     # plot fit and residual

  # Direction Independent Imaging Parameters
  di_img :
    source_ext  : ""        # str, extension to source name for image files

  # Direction Dependent Imaging Parameters
  dd_img :
    source_ext  : "_peel"        # str, extension to source name for image files

  # Direction Dependent Calibration Parameters
  dd_cal :
    # Parameters for source modeling
    model_ext       : "self_model"    # file extension for source model image cube
    pxsize          : 300             # pixel size in arcsec, must match input MFS images!
    imsize          : 512             # image side-length in pixels, must match input MFS images!
    stokes          : 'IQUV'          # stokes parameters to image, and therefore model
    inp_images      : "zen.2458101.29360.HH.uvR.ms.gleam02.spec????.image.fits" # glob-parseable string of input MFS images as coarse spectral cube
    sourcefile      : "clean_sources.tab" # source-file of source locations (see find_sources.py)
    rb_Npix         : 41              # side-length of restoring beam image in pixels
    fit_pl          : True            # bool, if True fit a power law to spectra
    fit_gp          : True            # bool, if True fit a GPR to the spectra
    gp_ls           : 5.0             # float, GP max length scale in MHz
    gp_nl           : 0.1             # float, GP noise level in spectra amplitude units
    exclude_sources : [0]             # list, list of source indices to exclude from model
    taper_alpha     : 0.1             # float, alpha parameter of tukey window applied to spectra
    search_frac     : 0.25            # float, PSF fraction within which to search for a source's peak flux
    # Parameters for new flux model
    file_ext        : "_peel"         # str, extension to gleam in output filename gleam{}.cl.fits, etc.
    regions         : ""
    region_radius   : 0.5             # float, radius in degrees around each clean source to exclude in new model. Should be ~PSF FWHM/2
    # Parameter for Calibration
    refant : 10           # int or comma-delimited str of reference antenna(s)
    rflag : False         # bool, run rflag task
    KGcal : False          # bool, run Delay and Mean Phase cal
    KGsnr : 0.0           # float, SNR cut in KGcal
    Acal  : False          # bool, run Mean Amp cal
    Asnr  : 0.0           # float, SNR cut in Acal
    BPcal : True          # bool, run Bandpass cal
    BPsnr : 0.0           # float, SNR cut in BPcal
    BPsolnorm : False     # bool, normalize BP amplitude to unity across freq
    uvrange   : ""        # str, UVrange in calibration
    timerange : ""        # str, UTC time-range to use in calibration
    gain_spw  : ""        # str, spw selection for gaincal calls
    bp_spw    : ""        # str, spw selection for bandpass call
    split_cal : False     # bool, split CORRECTED column from MS
    split_ext : "split"   # str, file extension of CORRECTED split
    split_model : True   # str, split MODEL column into a new .model MS
    gain_ext  : ''        # str, file extension of gain tables
    gaintables  : ''            # pre-calibration gain tables to apply first

