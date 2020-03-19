import os
import sys

from argparse import ArgumentParser, RawTextHelpFormatter
from astropy.io import fits
from distutils.version import LooseVersion
import math
import numpy as np
from scipy.ndimage.morphology import binary_dilation

# Check numpy and scipy version numbers for the nanmedian function import
if LooseVersion(np.__version__) >= LooseVersion("1.9.0"):
    from numpy import nanmedian
elif LooseVersion(sp.__version__) < LooseVersion("0.15.0"):
    from scipy.stats import nanmedian
else:
    from scipy import nanmedian


def make_param_file(sig=4, loc_dir=None, cube_name=None, cube=None):
    param_template = 'parameter_template_{}sig.par'.format(sig)
    new_paramfile = loc_dir + 'param_scTrel_{}sig.par'.format(sig)
    outlog = loc_dir + 'sourcefinding_{}sig.out'.format(sig)
    outroot = cube_name + '_{}sig_dev'.format(sig)

    # Edit parameter file (remove lines that need editing)
    os.system('grep -vwE "(input.data)" ' + param_template + ' > ' + new_paramfile)
    os.system('grep -vwE "(output.filename)" ' + new_paramfile + ' > temp && mv temp ' + new_paramfile)
    if cube == 3: os.system('grep -vwE "(flag.region)" ' + new_paramfile + ' > temp && mv temp ' + new_paramfile)

    # Add back the parameters needed
    os.system('echo "input.data                 =  ' + filteredfits + '" >> ' + new_paramfile)
    os.system('echo "output.filename            =  ' + outroot + '" >> ' + new_paramfile)
    if cube == 3: os.system('echo "flag.region                =  0,661,0,661,375,601" >> ' + new_paramfile)

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

# Parse the arguments above
args = parser.parse_args()

# Parse the arguments above
args = parser.parse_args()

# Range of cubes/beams to work on:
taskid = args.taskid
cubes = [int(c) for c in args.cubes.split(',')]
if '-' in args.beams:
    b_range = args.beams.split('-')
    beams = np.array(range(int(b_range[1])-int(b_range[0])+1)) + int(b_range[0])
else:
    beams = [int(b) for b in args.beams.split(',')]

# Parse the arguments above
args = parser.parse_args()


# Main source finding code for all cubes/beams
for b in beams:
    # Define some file names and work space:
    loc = '/tank/hess/apertif/' + taskid + '/B0' + str(b).zfill(2) + '/'

    for c in cubes:
        cube_name = 'HI_image_cube' + str(c)
        print("Working on Beam {:02} Cube {}".format(b, c))

        sourcefits = loc + cube_name + '.fits'
        filteredfits = loc + cube_name + '_filtered.fits'

        # Check to see if the continuum filtered file exists.  If not, make it  with SoFiA-2
        if os.path.isfile(filteredfits):
            print("\tContinuum filtered file exists.")
            with fits.open(filteredfits, mode='update') as f:
                # mask = np.ones(f[0].data.shape[0], dtype=bool)  # Commented out because doesn't seem to affect DR
                # if c == 3: mask[376:662] = False
                nanmax, nanstd = np.nanmax(f[0].data), np.nanstd(f[0].data)
                dynamic_range = nanmax / nanstd
                print("\tPsuedo dynamic range, max, std are: {}, {}, {}".format(dynamic_range, nanmax, nanstd))
        else:
            print("\tMaking continuum filtered file.")
            os.system('grep -vwE "(input.data)" template_filtering.par > ' + loc + 'filtering.par')
            os.system('echo "input.data                 =  ' + sourcefits + '" >> ' + loc + 'filtering.par')
            os.system('/home/apercal/SoFiA-2/sofia ' + loc + 'filtering.par >> test.log')
            with fits.open(filteredfits) as f:
                # mask = np.ones(f[0].data.shape[0], dtype=bool)  # Commented out because doesn't seem to affect DR
                # if c == 3: mask[376:662] = False
                nanmax, nanstd = np.nanmax(f[0].data), np.nanstd(f[0].data)
                dynamic_range = nanmax / nanstd
                print("\tPsuedo dynamic range, max, std are: {}, {}, {}".format(dynamic_range, nanmax, nanstd))

        # After filtering, if the DR (presumably from a bright HI source) is > 14 Do an initial source finding at sn=10.
        if (c != 1) & (dynamic_range >= 14.0):
            print("\tNEED TO DO HIGH THRESHOLD (10 sigma) SOURCE FINDING FIRST.")
            sig = 10
            new_paramfile, outlog = make_param_file(sig=sig, loc_dir=loc, cube_name=cube_name, cube=c)
            os.system('/home/apercal/SoFiA-2/sofia ' + new_paramfile + ' >> ' + outlog)
        else:
            print("\tDOING 4 sigma SOURCE FINDING.")
            sig = 4
            new_paramfile, outlog = make_param_file(sig=sig, loc_dir=loc, cube_name=cube_name, cube=c)
            os.system('/home/apercal/SoFiA-2/sofia ' + new_paramfile + ' >> ' + outlog)

        if (sig == 10) & (not os.path.isfile(loc + cube_name + '10sig_dev_cat.txt')):
            print("\t10 sigma found nothing. DOING 4 sigma SOURCE FINDING.")
            sig = 4
            new_paramfile, outlog = make_param_file(sig=sig, loc_dir=loc, cube_name=cube_name, cube=c)
            os.system('/home/apercal/SoFiA-2/sofia ' + new_paramfile + ' >> ' + outlog)

        # After source finding, reduce filtered cube to only one channel!  Need to work on expanding cube then in the first if/else.