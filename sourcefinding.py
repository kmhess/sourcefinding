import os

from src import checkmasks

from argparse import ArgumentParser, RawTextHelpFormatter
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
    os.system('echo "input.data                 =  ' + filteredfits + '" >> ' + new_paramfile)
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

        print("[SOURCEFINDING] Doing source finding with 4 sigma threshold.")
        sig = 4
        new_paramfile, outlog = make_param_file(sig=sig, loc_dir=loc, cube_name=cube_name, cube=c)
        os.system('/home/apercal/SoFiA-2/sofia ' + new_paramfile + ' >> ' + outlog)

    # After all cubes are done, run checkmasks to get summary plots for cleaning:
    checkmasks.main(taskid, [b])
