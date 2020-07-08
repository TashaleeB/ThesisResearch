#!/bin/sh

#  4pol_casa_imaging_pipeline.py
#  
#
#  Created by Tashalee Billings on 2/4/19.
#  
"""
    skycal_pipe.py
    -----------------------------------------
    This script was inspired by Nicholas Kern's automatic calibration
    and imaging pipeline, "casa_imaging_pipeline.py", in CASA for HERA data.
    
    Run as skycal_pipe.py -c skycal_params.yml <options>
    
    See skycal_params.yml for relevant parameter selections.

"""
import numpy as np
from pyuvdata import UVData
#import casa_imaging
#from casa_imaging import casa_utils as utils # what?? :(
import source2file
import pyuvdata.utils as uvutils
import os
import sys
import glob
import yaml
from datetime import datetime
import json
import itertools
import shutil
from collections import OrderedDict as odict
from astropy.time import Time
import copy
import operator
import subprocess
import argparse

#-------------------------------------------------------------------------------
# Define Optional Command-Line Parameters
#-------------------------------------------------------------------------------
args = argparse.ArgumentParser(description="skycal_pipe.py: run as python skycal_pipe.py -c skycal_params.yml <opts>")
args.add_argument("-c", "--param_file", type=str, help='Path to a YAML parameter file. See skycal_params.yml for details.')
# Optional Parameters that, if defined, overwrite their counterpart in param_file.yml
args.add_argument("--data_root", default=None, type=str, help="Root path to data files: overwrites skycal_params.yml")
args.add_argument("--data_file", default=None, type=str, help="Data file basename zen.2457548.46619.*.HH.uvcRP : overwrites skycal_params.yml")
args.add_argument("--file_type", default=None, type=str, help="Specify the file type that you are reading in using pyuvdata. Options are ['uvh5','uvfits','fhd','miriad']")
args.add_argument("--source", default=None, type=str, help="Source name: overwrites skycal_params.yml")
args.add_argument("--source_ra", default=None, type=float, help="Source right ascension in J2000 degrees: overwrites skycal_params.yml ")
args.add_argument("--source_dec", default=None, type=float, help="Source declination in J2000 degrees: overwrites skycal_params.yml")
a = args.parse_args()


def load_config(config_file):
    """
        Load configuration details from a YAML file.
        All entries of 'None' --> None and all lists
        of lists become lists of tuples.
        """
    # define recursive replace function
    def replace(d):
        if isinstance(d, (dict, odict)):
            for k in d.keys():
                # 'None' and '' turn into None
                if d[k] == 'None': d[k] = None
                # list of lists turn into lists of tuples
                if isinstance(d[k], list) and np.all([isinstance(i, list) for i in d[k]]):
                    d[k] = [tuple(i) for i in d[k]]
                elif isinstance(d[k], (dict, odict)): replace(d[k])

    # Open and read config file
    with open(config_file, 'r') as cfile:
        cfg = yaml.load(cfile)

    # Replace entries
    replace(cfg)

    return cfg

#-------------------------------------------------------------------------------
# Parse YAML Configuration File
#-------------------------------------------------------------------------------
# Get config and load dictionary
cf = utils.load_config(a.param_file)

# Consolidate IO, data and analysis parameter dictionaries
params = odict(cf['io'].items() + cf['obs'].items() + cf['data'].items() + cf['analysis'].items())
assert len(params) == len(cf['io']) + len(cf['obs']) + len(cf['data']) + len(cf['analysis']), ""\
    "Repeated parameters found within the scope of io, obs, data and analysis dicts"

# if optional argparser arguments passed, use their values
for v in ['data_root', 'data_file', 'source', 'source_ra', 'source_dec']:
    if getattr(a, v) is not None: # returns the value of the named attribute of an object
        params[v] = getattr(a, v) # returns a.v

# Get algorithm dictionary
algs = cf['algorithm']
datafile = os.path.join(params['data_root'], params['data_file'])

# Get parameters used globally in the pipeline
verbose = params['verbose']
overwrite = params['overwrite']
casa = params['casa'].split() + params['casa_flags'].split()
longitude = params['longitude']
latitude = params['latitude']
out_dir = params['out_dir']
source_ra = params['source_ra']
source_dec = params['source_dec']
source = params['source']

"""
# Change to working dir
os.chdir(params['work_dir'])

# Open a logfile
logfile = os.path.join(out_dir, params['logfile'])
if os.path.exists(logfile) and params['overwrite'] == False:
    raise IOError("logfile {} exists and overwrite == False, quitting pipeline...".format(logfile))
    lf = open(logfile, "w") #file descriptor to write to
if params['joinlog']:
    ef = lf
else:
    ef = open(os.path.join(params['out_dir'], params['errfile']), "w")
casa += ['--logfile', logfile]
"""

# Setup (Small) Global Variable Dictionary
varlist = ['datafile', 'verbose', 'overwrite', 'out_dir', 'casa', 'source_ra',
           'source_dec', 'source','longitude', 'latitude', 'lf', 'gaintables']

def global_vars(varlist=[]):
    d = []
    for v in varlist:
        try:
            d.append((v, globals()[v]))
        except KeyError:
            continue
    return dict(d)

# Print out parameter header
time = datetime.utcnow()
utils.log("Starting skycal_pipe.py on {}\n{}\n".format(time, '-'*60),verbose=verbose)
_cf = copy.copy(cf)
_cf.pop('algorithm')
utils.log(json.dumps(_cf, indent=1) + '\n', verbose=verbose) # prints and (in theory) writes to a file descriptor log.

# Setup a dict->object converter
class Dict2Obj:
    def __init__(self, **entries):
        self.__dict__.update(entries)

#-------------------------------------------------------------------------------
# Search for a Source and Prepare Data for MS Conversion
#-------------------------------------------------------------------------------
if params['prep_data']:
    # start block
    time = datetime.utcnow()
    utils.log("\n{}\n...Starting PREP_DATA: {}\n".format("-"*60, time),
              verbose=verbose)
    utils.log(json.dumps(algs['prep_data'], indent=1) + '\n', verbose=verbose)
    p = Dict2Obj(**algs['prep_data'])

    # Check if datafile is already MS
    
    if os.path.splitext(datafile)[1] in ['.ms', '.MS']:
        # get transit time of source
        if source_ra is not None:
            (lst, transit_jd, utc_range, utc_center, source_files,
             source_utc_range) = source2file.source2file(source_ra, filetype=a.file_type, lon=longitude, lat=latitude,duration=p.duration, start_jd=p.start_jd, get_filetimes=False,verbose=verbose)
            utils.log("...the file {} is already a CASA MS, skipping rest of PREP_DATA".format(datafile),  verbose=verbose)
            timerange = utc_range
        else:
            transit_jd = None

    else:
        # Iterate over polarizations
        if p.pols is None: p.pols = [None]
        uvds = []
        for pol in p.pols:
            if pol is None:
                pol = ''
            else:
                utils.log("...working on {} polarization".format(pol), verbose=verbose)
                pol = '.{}.'.format(pol)
    
            # glob-parse the data file / template
            datafiles = [df for df in glob.glob(datafile) if pol in df]
            assert len(datafiles) > 0, "Searching for {} with pol {} but found no files...".format(datafile, pol)
            
            # get transit times
            if source_ra is not None:
                (lst, transit_jd, utc_range, utc_center, source_files,
                 source_utc_range) = source2file.source2file(source_ra, filetype=a.file_type, lon=longitude, lat=latitude,
                                                             duration=p.duration, start_jd=p.start_jd, get_filetimes=p.get_filetimes,
                                                             verbose=verbose, jd_files=copy.copy(datafiles))
                timerange = utc_range
                                                             
                # ensure source_utc_range and utc_range are similar
                if source_utc_range is not None:
                    utc_range_start = utc_range.split('~')[0].strip('"').split('/')
                    utc_range_start = map(int, utc_range_start[:-1] + utc_range_start[-1].split(':'))
                    utc_range_start = Time(datetime(*utc_range_start), format='datetime').jd
                    source_utc_range_start = source_utc_range.split('~')[0].strip('"').split('/')
                    source_utc_range_start = map(int, source_utc_range_start[:-1] + source_utc_range_start[-1].split(':'))
                    source_utc_range_start = Time(datetime(*source_utc_range_start), format='datetime').jd
                    # if difference is larger than 1 minute,
                    # then probably the correct files were not found
                    if np.abs(utc_range_start - source_utc_range_start) * 24 * 60 > 1:
                        utils.log("Warning: Difference between theoretical transit time and transit time " \
                                                                               "deduced from files found is larger than 1-minute: probably the correct " \
                                                                               "files were not found because the correct files did not exist under the " \
                                                                               "data template {}".format(datafile), verbose=verbose)
                    timerange = source_utc_range
            else:
                source_files = datafiles
        
            # load data into UVData
            utils.log("...loading data", verbose=verbose)
            _uvds = []
            for sf in list(source_files):
                # read data
                _uvd = UVData()
                _uvd.read(sf, file_type=a.file_type, antenna_nums=p.antenna_nums)
                
                # read flagfile if fed
                if p.flag_ext != "":
                    flagfile = glob.glob("{}{}".format(sf, p.flag_ext))
                    if len(flagfile) == 1:
                        utils.log("...loading and applying flags {}".format(flagfile[0]), verbose=verbose)
                        ff = np.load(flagfile[0])
                        _uvd.flag_array += ff['flag_array']
        
                # append to list
                _uvds.append(_uvd)
            
            # concatenate source files
            uvd = reduce(operator.add, _uvds)
            
            # isolate only relevant times
            times = np.unique(uvd.time_array)
            if source_ra is not None:
                transit_jd = np.mean(times)
            times = times[np.abs(times - transit_jd) < (p.duration / (24. * 60. * 2))]
            assert len(times) > 0, "No times found in source_files {} given transit JD {} and duration {}".format(source_files, transit_jd, p.duration)
            uvd.select(times=times)
            
            # append
            uvds.append(uvd)

        # concatenate uvds
        uvd = reduce(operator.add, uvds)

        # get output filepath w/o uvfits extension if provided
        outfile = os.path.join(params['out_dir'], p.outfile.format(uvd.time_array.min()))
        if os.path.splitext(outfile)[1] == '.uvfits':
            outfile = os.path.splitext(outfile)[0]
        
        # renumber antennas (and antenna names!) if above 254
        if uvd.antenna_numbers.max() > 254:
            large_ant_nums = sorted(list(uvd.antenna_numbers[np.where(uvd.antenna_numbers > 254)[0]]))
            new_nums = sorted(list(set(range(255)) - set(uvd.antenna_numbers)))
            if len(new_nums) < len(large_ant_nums):
                raise ValueError('too many antennas in dataset, cannot renumber all below 255')
            new_nums = new_nums[-1 * len(large_ant_nums):]
            renumber_dict = dict(list(zip(large_ant_nums, new_nums)))
            
            history = ''
            name_prefix = os.path.commonprefix(uvd.antenna_names)
            for ant_in, ant_out in renumber_dict.items():
                if verbose:
                    msg = "renumbering {a1} to {a2}".format(a1=ant_in, a2=ant_out)
                    print(msg)
                history += '{}\n'.format(msg)
                
                wh_ant_num = np.where(uvd.antenna_numbers == ant_in)[0]
                wh_ant1_arr = np.where(uvd.ant_1_array == ant_in)[0]
                wh_ant2_arr = np.where(uvd.ant_2_array == ant_in)[0]
                
                uvd.antenna_numbers[wh_ant_num] = ant_out
                uvd.antenna_names[wh_ant_num[0]] = "RN{:d}".format(ant_out)
                uvd.ant_1_array[wh_ant1_arr] = ant_out
                uvd.ant_2_array[wh_ant2_arr] = ant_out
            
            uvd.baseline_array = uvd.antnums_to_baseline(uvd.ant_1_array, uvd.ant_2_array)
            uvd.history = '{}\n{}'.format(history, uvd.history)
            uvd.check()
            
            # write renumbering dictionary to .npz
            np.savez("{}.renumber.npz".format(outfile),
                     renumber=dict(zip(renumber_dict.values(), renumber_dict.keys())),
                     history="Access dictionary via f['renumber'].item()")
        
        # write to file
        if uvd.phase_type == 'phased':
            # write uvfits
            uvfits_outfile = outfile + '.uvfits'
            if not os.path.exists(uvfits_outfile) or overwrite:
                utils.log("...writing {}".format(uvfits_outfile), verbose=verbose)
                uvd.write_uvfits(uvfits_outfile, spoof_nonessential=True)
            # unphase to drift
            uvd.unphase_to_drift()
            # write uvh5
            if not os.path.exists(outfile+'.uvh5') or overwrite:
                utils.log("...writing {}".format(outfile+'.uvh5'),verbose=verbose)
                uvd.write_uvh5(outfile +'.uvh5', clobber=True)
        elif uvd.phase_type == 'drift':
            # write uvh5
            if not os.path.exists(outfile+'.uvh5') or overwrite:
                utils.log("...writing {}".format(outfile+'.uvh5'), verbose=verbose)
                uvd.write_uvh5(outfile +'.uvh5', clobber=True)
            # write uvfits
            uvfits_outfile = outfile + '.uvfits'
            if not os.path.exists(uvfits_outfile) or overwrite:
                uvd.phase_to_time(Time(transit_jd, format='jd'))
                utils.log("...writing {}".format(uvfits_outfile), verbose=verbose)
                uvd.write_uvfits(uvfits_outfile, spoof_nonessential=True)

        # convert the uvfits file to ms
        ms_outfile = outfile + '.MS'
        utils.log("...converting to Measurement Set")
        if not os.path.exists(ms_outfile) or overwrite:
            if os.path.exists(ms_outfile):
                shutil.rmtree(ms_outfile)
            utils.log("...writing {}".format(ms_outfile), verbose=verbose)
            ecode = subprocess.check_call(casa + ["-c", "importuvfits('{}', '{}')".format(uvfits_outfile, ms_outfile)])
    
        # overwrite relevant parameters for downstream analysis
        datafile = ms_outfile

        del uvds, uvd

    # overwrite downstream parameters
    algs['gen_model']['time'] = transit_jd

    # end block
    time2 = datetime.utcnow()
    utils.log("...finished PREP_DATA: {:d} sec elapsed".format(utils.get_elapsed_time(time, time2)), verbose=verbose)

#-------------------------------------------------------------------------------
# Generate Flux Model
#-------------------------------------------------------------------------------

# Make a Model Generation Function
def gen_model(**kwargs):
    p = Dict2Obj(**kwargs) #dict->object converter
    utils.log("\n{}\n...Generating a Flux Model", verbose=p.verbose)
    
    # compile complist_gleam.py command
    cmd = casa + ["-c", "{}/complist_gleam.py".format(casa_scripts)]
    cmd += ['--point_ra', p.source_ra, '--point_dec', p.latitude, '--outdir', p.out_dir,
            '--gleamfile', p.gleamfile, '--radius', p.radius, '--min_flux', p.min_flux,
            '--freqs', p.freqs, '--cell', p.cell, '--imsize', p.imsize]
    if p.image:
        cmd += ['--image']
    if p.use_peak:
        cmd += ['--use_peak']
    if p.overwrite:
        cmd += ['--overwrite']
    if hasattr(p, 'regions'):
        cmd += ['--regions', p.regions, '--exclude', '--region_radius', p.region_radius]
    if hasattr(p, 'file_ext'):
        cmd += ['--ext', p.file_ext]
    else:
        p.file_ext = ''
    cmd = map(str, cmd)
    ecode = subprocess.check_call(cmd)

    modelstem = os.path.join(p.out_dir, "gleam{}.cl".format(p.file_ext))
    model = modelstem
    if p.image:
        model += ".image"

    # pbcorrect
    if p.pbcorr:
        utils.log("...applying PB to model", f=p.lf, verbose=p.verbose)
        assert p.image, "Cannot pbcorrect flux model without image == True"
        cmd = ["pbcorr.py", "--lon", p.longitude, "--lat", p.latitude, "--time", p.time, "--pols"] \
            + [uvutils.polstr2num(pol) for pol in p.pols] \
            + ["--outdir", p.out_dir, "--multiply", "--beamfile", p.beamfile]
        if p.overwrite:
            cmd.append("--overwrite")
        cmd.append(modelstem + '.fits')

        # generate component list and / or image cube flux model
        cmd = map(str, cmd)
        ecode = subprocess.check_call(cmd)
        modelstem = os.path.join(p.out_dir, modelstem)

        # importfits
        cmd = p.casa + ["-c", "importfits('{}', '{}', overwrite={})".format(modelstem + '.pbcorr.fits', modelstem + '.pbcorr.image', p.overwrite)]
        ecode = subprocess.check_call(cmd)
        model = modelstem + ".pbcorr.image"
    
    return model

if params['gen_model']:
    # start block
    time = datetime.utcnow()
    utils.log("\n{}\n...Starting GEN_MODEL: {}\n".format("-"*60, time),verbose=verbose)
    utils.log(json.dumps(algs['gen_model'], indent=1) + '\n', verbose=verbose)

    # Generate Model
    model = gen_model(**dict(algs['gen_model'].items() + global_vars(varlist).items()))

    # update di_cal model path
    algs['di_cal']['model'] = model

    # end block
    time2 = datetime.utcnow()
    utils.log("...finished GEN_MODEL: {:d} sec elapsed".format(utils.get_elapsed_time(time, time2)), verbose=verbose)


