#!/usr/bin/env python

from __future__ import print_function, division, absolute_import
import numpy as np
import re
import h5py
import argparse

ap = argparse.ArgumentParser(description="Convert .npz-saved spectrum to hdf5")
ap.add_argument("--fname", type=str, help="name of .npz file", required=True)
ap.add_argument("--outname", type=str, default=None, help="name of output.hdf5 file. Default is input "
                "file with .npz extension changed to .h5")
ap.add_argument("--npzkey", type=str, help="name of the key in .npz file.",required=True)
ap.add_argument("--h5key", type=str, help="name of key for .hdf5 file.", required=True)

def main(args):
    fname = args.fname
    npf = np.load(fname)
    spec = npf[args.npzkey]
    if args.outname:
        outname = args.outname
    else:
        outname = re.sub('\.npz', '.h5', fname)
    with h5py.File(outname, 'a') as f:
        f['Data/'+args.h5key] = spec

if __name__ == '__main__':
    args = ap.parse_args()
    main(args)
