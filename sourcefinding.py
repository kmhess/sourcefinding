import os

from modules.natural_cubic_spline import fspline
from src import checkmasks

from argparse import ArgumentParser, RawTextHelpFormatter
from astropy.io import fits
import numpy as np


def make_param_file(sig=4, loc_dir=None, cube_name=None, cube=None):
    param_template = 'parameter_template_{}sig.par'.format(sig)
    new_paramfile = loc_dir + 'parameter_{}sig.par'.format(sig)
    outlog = loc_dir + 'sourcefinding_{}sig.out'.format(sig)
    outroot = cube_name + '_{}sig'.format(sig)

    # Edit parameter file (remove lines that need editing)
    os.system('grep -vwE "(input.data)" ' + param_template + ' > ' + new_paramfile)
    os.system('grep -vwE "(output.filename)" ' + new_paramfile + ' > temp && mv temp ' + new_paramfile)
    if cube == 3:
        os.system('grep -vwE "(flag.region)" ' + new_paramfile + ' > temp && mv temp ' + new_paramfile)

    # Add back the parameters needed
    os.system('echo "input.data                 =  ' + splinefits + '" >> ' + new_paramfile)
    os.system('echo "output.filename            =  ' + outroot + '" >> ' + new_paramfile)
    if cube == 3:
        os.system('echo "flag.region                =  0,661,0,661,375,601" >> ' + new_paramfile)

    return new_paramfile, outlog


###################################################################

parser = ArgumentParser(description="Do source finding in the HI spectral line cubes for a given taskid, beam, cubes",
                        formatter_class=RawTextHelpFormatter)

parser.add_argument('-t', '--taskid', default='190915041',
                    help='Specify the input taskid (default: %(default)s).')

parser.add_argument('-b', '--beams', default='0-39',
                    help='Specify a range (0-39) or list (3,5,7,11) of beams on which to do source finding (default: %(default)s).')

parser.add_argument('-c', '--cubes', default='1,2,3',
                    help='Specify the cubes on which to do source finding (default: %(default)s).')

parser.add_argument('-o', "--overwrite",
                    help="If option is included, overwrite old continuum filtered file if it exists.",
                    action='store_true')

# Parse the arguments above
args = parser.parse_args()

# Range of cubes/beams to work on:
taskid = args.taskid
cubes = [int(c) for c in args.cubes.split(',')]
if '-' in args.beams:
    b_range = args.beams.split('-')
    beams = np.array(range(int(b_range[1]) - int(b_range[0]) + 1)) + int(b_range[0])
else:
    beams = [int(b) for b in args.beams.split(',')]
overwrite = args.overwrite

# Main source finding code for all cubes/beams
for b in beams:
    # Define some file names and work space:
    loc = '/tank/hess/apertif/' + taskid + '/B0' + str(b).zfill(2) + '/'

    for c in cubes:
        cube_name = 'HI_image_cube' + str(c)
        print("[SOURCEFINDING] Working on Beam {:02} Cube {}".format(b, c))

        sourcefits = loc + cube_name + '.fits'
        filteredfits = loc + cube_name + '_filtered.fits'
        splinefits = loc + cube_name + '_spline.fits'
        # Output exactly where sourcefinding is starting
        print('\t' + sourcefits)

        # Check to see if the continuum filtered file exists.  If not, make it  with SoFiA-2
        if (not overwrite) & os.path.isfile(filteredfits):
            print("[SOURCEFINDING] Continuum filtered file exists and will not be overwritten.")
        elif os.path.isfile(sourcefits):
            print("[SOURCEFINDING] Making continuum filtered file.")
            os.system('grep -vwE "(input.data)" template_filtering.par > ' + loc + 'filtering.par')
            os.system('echo "input.data                 =  ' + sourcefits + '" >> ' + loc + 'filtering.par')
            os.system('/home/apercal/SoFiA-2/sofia ' + loc + 'filtering.par >> test.log')
        else:
            print("\tBeam {:02} Cube {} is not present in this directory.".format(b, c))
            continue

        if (not overwrite) & os.path.isfile(splinefits):
            print("[SOURCEFINDING] Spline fitted file exists and will not be overwritten.")
        elif os.path.isfile(sourcefits):
            print("[SOURCEFINDING] Making spline fitted file.")
            os.system('cp {} {}'.format(sourcefits, splinefits))
            splinecube = fits.open(splinefits, mode='update')
            orig = fits.open(sourcefits)
            # Try masking strong sources to not bias fit
            mask = 2.5 * np.nanstd(orig[0].data)
            splinecube[0].data[np.abs(splinecube[0].data) >= mask] = np.nan

            # Do the spline fitting on the z-axis to masked cube, replace spline cube with original minus fit
            for x in range(orig[0].data.shape[1]):
                print(x)
                for y in range(orig[0].data.shape[2]):
                    fit = fspline(np.linspace(1, orig[0].data.shape[0], orig[0].data.shape[0]),
                                  np.nan_to_num(splinecube[0].data[:, x, y]), k=5)
                    splinecube[0].data[:, x, y] = orig[0].data[:, x, y] - fit

            splinecube.flush()
            orig.close()

        print("[SOURCEFINDING] Doing source finding with 4 sigma threshold.")
        sig = 4
        new_paramfile, outlog = make_param_file(sig=sig, loc_dir=loc, cube_name=cube_name, cube=c)
        os.system('/home/apercal/SoFiA-2/sofia ' + new_paramfile + ' >> ' + outlog)

    # After all cubes are done, run checkmasks to get summary plots for cleaning:
    checkmasks.main(taskid, [b])
