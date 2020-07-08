from __future__ import print_function, division, absolute_import
import argparse,os

a = argparse.ArgumentParser(description="Run with casa as: casa -c CASA_imager.py <args>")
a.add_argument('--script', '-c', type=str, help='name of this script', required=True)
a.add_argument('--file_path', '-fp', type=str, help='Path to file. eg. "/lustre/aoc/projects/hera/jaguirre/PolarizedMosaics/"',
               required=True)

if __name__ == '__main__':
    args = a.parse_args()
    
    text_file_prefix = args.file_path

    summary = open(text_file_prefix+".phase_center_per_LST_slice.txt", "r")
    fitsfile, ra0, dec0 = [x[:-1] for x in summary.readlines()]
    
    # Make CASA MS
    print("Making Measurement Set, ",fitsfile+".MS")
    msfile = fitsfile+".MS"
    importuvfits(fitsfile, msfile)
    os.system("rm -rvf "+ fitsfile)
    
    # Make Dirty Image and Deconvolved Image of the MS.
    tclean(vis=msfile, imagename=text_file_prefix+'.no_deconvolution', niter=0,
           weighting='briggs',robust=0,imsize = [512,512], pbcor=False, cell=['500 arcsec'],
           specmode='mfs', nterms=1, spw='0:100~920',stokes='IQUV', interactive=False, pblimit=-1,
           phasecenter='J2000 %sdeg %sdeg' % (ra0, dec0))
    viewer(infile=text_file_prefix+'.no_deconvolution.image',
           outformat=text_file_prefix+'.no_deconvolution.image.pdf', gui=False)
    
    tclean(vis=msfile, imagename=text_file_prefix+'.deconvolved', niter=5000,
           weighting='briggs', robust=0, imsize = [512,512], pbcor=False, cell=['500 arcsec'],
           specmode='mfs', nterms=1, spw='0:100~920', stokes='IQUV', mask=text_file_prefix+'.masks.txt',
           interactive=False, cycleniter=1000, threshold='1Jy/beam',pblimit=-1,
           phasecenter='J2000 %sdeg %sdeg' % (ra0, dec0))
    viewer(infile=text_file_prefix+'.deconvolved.image',
           outformat=text_file_prefix+'.deconvolved.image.pdf', gui=False)
    
    print("Exporting CASA image files to FITZ files.")
    exportfits(text_file_prefix+".no_deconvolution.image", text_file_prefix+".no_deconvolution.image.fits")
    exportfits(text_file_prefix+".deconvolved.image", text_file_prefix+".deconvolved.image.fits")
    os.system("rm -rvf *.log *.last")
