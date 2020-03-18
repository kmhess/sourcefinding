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


# For SoFiA-1 continuum filtering (do not edit)
threshold = 4.0
dilation = 3

# ==========================================
# Do continuum filtering (stolen from SoFiA)
# ==========================================

def flaglos(cube, threshold, dilation):
    rms0 = GetRMS(cube, rmsMode="mad", fluxRange="negative", zoomx=1, zoomy=1, zoomz=1, verbose=0, twoPass=True)

    los_rms = np.zeros((cube.shape[-2], cube.shape[-1]))
    for xx in range(cube.shape[-1]):
        for yy in range(cube.shape[-2]):
            los_rms[yy, xx] = GetRMS(cube[:, yy:yy+1, xx:xx+1], rmsMode="mad", fluxRange="all", twoPass=True)

    # Mask all LOS whose RMS is > threshold*STD above rms0 (optionally extended to neighbouring LOS using binary dilation with a box structuring element)
    los_rms_disp = np.nanstd(los_rms)
    los_rms = (los_rms < rms0 + threshold * los_rms_disp)
    los_rms = binary_dilation(~los_rms, structure=np.ones((dilation, dilation)))
    cube[:, los_rms] = np.nan

    return cube


# ===================================
# Function to measure RMS noise level
# ===================================
def GetRMS(cube, rmsMode="negative", fluxRange="all", zoomx=1, zoomy=1, zoomz=1, verbose=0, min_hist_peak=0.05,
           sample=1, twoPass=False):
    """
    Description of arguments
    ------------------------
    rmsMode    Select which algorithm should be used for calculating the noise.
               Allowed options:
                 'mad'       Median absolute deviation about 0.
    fluxRange  Define which part of the data are to be used in the noise measurement.
               Allowed options:
                 'negative'  Use only pixels with negative flux.
                 'positive'  Use only pixels with positive flux.
                 'all'       Use both positive and negative (i.e. all) pixels.
    verbose    Print additional progress messages if set to True.
    twoPass    Run a second pass of MAD and STD, this time with a clip level of 5 times
               the RMS from the first pass.
    """

    # Check input for sanity
    if fluxRange != "all" and fluxRange != "positive" and fluxRange != "negative":
        sys.stderr.write("WARNING: Illegal value of fluxRange = '" + str(fluxRange) + "'.\n")
        sys.stderr.write("         Using default value of 'all' instead.\n")
        fluxRange = "all"
    if rmsMode != "std" and rmsMode != "mad" and rmsMode != "negative" and rmsMode != "gauss" and rmsMode != "moment":
        sys.stderr.write("WARNING: Illegal value of rmsMode = '" + str(rmsMode) + "'.\n")
        sys.stderr.write("         Using default value of 'mad' instead.\n")
        rmsMode = "mad"

    # Ensure that we have a 3D cube
    if len(cube.shape) == 2: cube = np.array([cube])

    x0, x1 = int(math.ceil((1 - 1.0 / zoomx) * cube.shape[2] / 2)), int(
        math.floor((1 + 1.0 / zoomx) * cube.shape[2] / 2)) + 1
    y0, y1 = int(math.ceil((1 - 1.0 / zoomy) * cube.shape[1] / 2)), int(
        math.floor((1 + 1.0 / zoomy) * cube.shape[1] / 2)) + 1
    z0, z1 = int(math.ceil((1 - 1.0 / zoomz) * cube.shape[0] / 2)), int(
        math.floor((1 + 1.0 / zoomz) * cube.shape[0] / 2)) + 1
    message("    Estimating rms on subcube (x,y,z zoom = %.0f,%.0f,%.0f) ..." % (zoomx, zoomy, zoomz), verbose)
    message("    Estimating rms on subcube sampling every %i voxels ..." % (sample), verbose)
    message("    ... Subcube shape is " + str(cube[z0:z1:sample, y0:y1:sample, x0:x1:sample].shape) + " ...",
                verbose)

    # Check if only negative or positive pixels are to be used:
    if fluxRange == "negative":
        with np.errstate(invalid="ignore"):
            halfCube = cube[z0:z1:sample, y0:y1:sample, x0:x1:sample][
                cube[z0:z1:sample, y0:y1:sample, x0:x1:sample] < 0]
        ensure(halfCube.size,
                   "Cannot measure noise from negative flux values.\nNo negative fluxes found in data cube.")
    elif fluxRange == "positive":
        with np.errstate(invalid="ignore"):
            halfCube = cube[z0:z1:sample, y0:y1:sample, x0:x1:sample][
                cube[z0:z1:sample, y0:y1:sample, x0:x1:sample] > 0]
        ensure(halfCube.size,
                   "Cannot measure noise from positive flux values.\nNo positive fluxes found in data cube.")
    # NOTE: The purpose of the with... statement is to temporarily disable certain warnings, as otherwise the
    #       Python interpreter would print a warning whenever a value of NaN is compared to 0. The comparison
    #       is defined to yield False, which conveniently removes NaNs by default without having to do that
    #       manually in a separate step, but the associated warning message is unfortunately a nuisance.

    # MEDIAN ABSOLUTE DEVIATION
    if rmsMode == "mad":
        if fluxRange == "all":
            # NOTE: Here we assume that the median of the data is zero!
            rms = 1.4826 * nanmedian(abs(cube[z0:z1:sample, y0:y1:sample, x0:x1:sample]), axis=None)
            if twoPass:
                message("Repeating noise estimation with 5-sigma clip.", verbose)
                with np.errstate(invalid="ignore"):
                    rms = 1.4826 * nanmedian(abs(cube[z0:z1:sample, y0:y1:sample, x0:x1:sample][
                                                     abs(cube[z0:z1:sample, y0:y1:sample, x0:x1:sample]) < 5.0 * rms]),
                                             axis=None)
        else:
            # NOTE: Here we assume that the median of the data is zero! There are no more NaNs in halfCube.
            rms = 1.4826 * np.median(abs(halfCube), axis=None)
            if twoPass:
                message("Repeating noise estimation with 5-sigma clip.", verbose)
                rms = 1.4826 * np.median(abs(halfCube[abs(halfCube) < 5.0 * rms]), axis=None)

    else:
        message("   ***RMSMODE MUST BE MAD***  Probably crashing out now...")

    message("    ... %s rms = %.2e (data units)" % (rmsMode, rms), verbose)

    return rms


# =====================================
# FUNCTION: Print informational message
# =====================================

def message(message, verbose=True):
    if verbose:
        sys.stdout.write(str(message) + "\n")
        sys.stdout.flush()
    return


# =========================================================
# FUNCTION: Check condition and exit with signal 1 if false
# =========================================================

def ensure(condition, message, fatal=True, frame=False):
    if not condition:
        error(message, fatal=fatal, frame=frame)
    return


def make_param_file(sig=4, loc_dir=None, cube_name=None, cube=None):
    param_template = 'parameter_template_{}sig.par'.format(sig)
    new_paramfile = loc_dir + 'param_scTrel_{}sig.par'.format(sig)
    outlog = loc_dir + 'sourcefinding_{}sig.out'.format(sig)
    outroot = cube_name + '_{}sig'.format(sig)

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
#cubes = args.cubes  # [1, 2, 3]  # Most sources in 2; nearest galaxies in 3.
if '-' in args.beams:
    b_range = args.beams.split('-')
    beams = np.array(range(int(b_range[1])-int(b_range[0])+1)) + int(b_range[0])
else:
    beams = [int(b) for b in args.beams.split(',')]
#beams = args.beams  # range(40)

# Parse the arguments above
args = parser.parse_args()


# Main source finding code for all cubes/beams
for c in cubes:
    cube_name = 'HI_image_cube' + str(c)
    for b in beams:
        print("Working on Cube {}, Beam {:02}".format(c, b))
        # Define some file names and work space:
        loc = '/tank/hess/apertif/' + taskid + '/B0' + str(b).zfill(2) + '/'

        sourcefits = loc + cube_name + '.fits'
        filteredfits = loc + cube_name + '_filtered_sof1.fits'

        # Check to see if the continuum filtered file exists.  If not, make it.
        if os.path.isfile(filteredfits):
            print("\tContinuum filtered file exists.")
            with fits.open(filteredfits, mode='update') as f:
                # mask = np.ones(f[0].data.shape[0], dtype=bool)  # Commented out because doesn't seem to affect DR
                # if c == 3: mask[376:662] = False
                nanmax, nanstd = np.nanmax(f[0].data), np.nanstd(f[0].data)
                dynamic_range = nanmax / nanstd
                print("\tPsuedo dynamic range, max, std are: {}, {}, {}".format(dynamic_range, nanmax, nanstd))
        else:
            # Copy cube
            os.system('cp ' + sourcefits + ' ' + filteredfits)
            # Read in cube
            print("\tMaking continuum filtered file.")
            with fits.open(filteredfits, mode='update') as f:
                pre_filter_data = f[0].data
                dict_Header = f[0].header
                # Do the filtering using SoFiA-1 functions above
                f[0].data = flaglos(pre_filter_data, threshold, dilation)
                # mask = np.ones(f[0].data.shape[0], dtype=bool)  # Commented out because doesn't seem to affect DR
                # if c == 3: mask[376:662] = False
                nanmax, nanstd = np.nanmax(f[0].data), np.nanstd(f[0].data)
                dynamic_range = nanmax / nanstd
                print("\tPsuedo dynamic range, max, std are: {}, {}, {}".format(dynamic_range, nanmax, nanstd))
                # Write out cube
                f.flush()

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

        if (sig == 10) & (not os.path.isfile(loc + cube_name + '10sig_cat.txt')):
            print("\t10 sigma found nothing. DOING 4 sigma SOURCE FINDING.")
            sig = 4
            new_paramfile, outlog = make_param_file(sig=sig, loc_dir=loc, cube_name=cube_name, cube=c)
            os.system('/home/apercal/SoFiA-2/sofia ' + new_paramfile + ' >> ' + outlog)

        # Do source finding
        # os.system('/home/apercal/SoFiA-2/sofia ' + new_paramfile + ' >> ' + outlog)
        # os.system('tac ' + outlog + '| grep -m 3 ound')

        # After source finding, reduce filtered cube to only one channel!  Need to work on expanding cube then in the first if/else.