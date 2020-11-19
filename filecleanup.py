import os

from astropy.io import fits
from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np


def make_2d_filtered_file():
    """
    Make a 2D file of the filtered pixel columns from either filtered.fits or filtered_spline.fits.
    Purpose is to shrink file sizes so final data products are more manageable.
    """
    return

def remake_filtered_cube():
    """
    Remake a filtered cube from filtered-2d.fits and original data cube
    Potentially not useful since SoFiA is very fast generating a filtered cube (at least when run in parallel). TBD.
    Potentially useful if user doesn't have SoFiA?
    """
    return

def remake_filtered_spline_cube():
    """
    Remake a filtered_spline cube from filtered-2d.fits and all_spline.fits.
    Useful for redoing source finding with different parameters.
    """
    return


parser = ArgumentParser(description="Do source finding in the HI spectral line cubes for a given taskid, beam, cubes",
                        formatter_class=RawTextHelpFormatter)

parser.add_argument('-t', '--taskid', default='190915041',
                    help='Specify the input taskid (default: %(default)s).')

parser.add_argument('-b', '--beams', default='0-39',
                    help='Specify a range (0-39) or list (3,5,7,11) of beams on which to do source finding (default: %(default)s).')

parser.add_argument('-c', '--cubes', default='1,2,3',
                    help='Specify the cubes on which to do source finding (default: %(default)s).')

parser.add_argument('-p', '--path', default='/tank/hess/apertif/',
                    help='Specify the directory path where the taskid folder lives (default: %(default)s).')

# Parse the arguments above
args = parser.parse_args()

print("\n********************************************************************************************************\n")
# Range of cubes/beams to work on:
taskid = args.taskid
cubes = [int(c) for c in args.cubes.split(',')]
if '-' in args.beams:
    b_range = args.beams.split('-')
    beams = np.array(range(int(b_range[1]) - int(b_range[0]) + 1)) + int(b_range[0])
else:
    beams = [int(b) for b in args.beams.split(',')]

print("[WARNING]: Only run this program at the end of all source finding activities on a given taskid/beam/cube !!!")

for b in beams:
    # Define some file names and work space:
    loc = args.path + taskid + '/B0' + str(b).zfill(2) + '/'

    for c in cubes:
        cube_name = 'HI_image_cube' + str(c)
        beam_name = 'HI_beam_cube' + str(c)
        filter_name = loc + cube_name + "_filtered.fits"
        filter2d_name = loc + cube_name + "_filtered-2d.fits"

        if os.path.isfile(loc + cube_name + '_4sig_mask-2d.fits') & (not os.path.isfile(filter2d_name)):
            print("[FILECLEANUP] Making {}".format(filter2d_name))
            os.system("cp " + loc + cube_name + "_4sig_mask-2d.fits " + filter2d_name)
            # Open the necessary files
            hdu_filter2d = fits.open(filter2d_name, mode = 'update')
            hdu_filter = fits.open(filter_name)
            # Assign a positive value to the filter and set everything else to nan:
            filter2d = np.full(hdu_filter2d[0].data.shape, np.nan)
            filter2d[np.isnan(hdu_filter[0].data[1, :, :])] = 1.
            hdu_filter2d[0].data = filter2d
            hdu_filter2d.flush()
            hdu_filter2d.close()
            hdu_filter.close()

        if os.path.isfile(loc + cube_name + '_filtered-2d.fits') & os.path.isfile(loc + cube_name + '_all_spline.fits'):
            print("\tSources were cleaned for {}{}.fits and filter-2d generated; deleting extra files".format(loc, cube_name))
            print("[FILECLEANUP] Deleting {}".format(loc + beam_name + ".fits"))
            os.system("rm -rf " + loc + beam_name + ".fits")
            print("[FILECLEANUP] Deleting {}".format(loc + cube_name + ".fits"))
            os.system("rm -rf " + loc + cube_name + ".fits")
            print("[FILECLEANUP] Deleting {}".format(loc + cube_name + "_filtered.fits"))
            os.system("rm -rf " + loc + cube_name + "_filtered.fits")
            print("[FILECLEANUP] Deleting {}".format(loc + cube_name + "_filtered_spline.fits"))
            os.system("rm -rf " + loc + cube_name + "_filtered_spline.fits")
            print("[FILECLEANUP] Deleting {}".format(loc + cube_name + "*_clean_cb.fits"))
            os.system("rm -rf " + loc + cube_name + "*_clean_cb.fits")

        elif (os.path.isfile(loc + cube_name + '_filtered-2d.fits') & (not os.path.isfile(loc + cube_name + '_all_spline.fits'))) &\
                (os.path.isfile(loc + cube_name + '_4sig_rel.eps')):
            print("\tNo sources cleaned for {}{}.fits; filter-2d generated; deleting extra files".format(loc, cube_name))
            print("[FILECLEANUP] Deleting {}".format(loc + cube_name + ".fits"))
            os.system("rm -rf " + loc + cube_name + ".fits")
            print("[FILECLEANUP] Deleting {}".format(loc + cube_name + "_filtered.fits"))
            os.system("rm -rf " + loc + cube_name + "_filtered.fits")
            print("[FILECLEANUP] Deleting {}".format(loc + cube_name + "_4sig_mask.fits"))
            os.system("rm -rf " + loc + cube_name + "_4sig_mask.fits")
            # Maybe get rid of _4sig_mask-2d.fits and the catalgg as well????  Hold off for now...
            # print("[FILECLEANUP] Deleting {}".format(loc + cube_name + "_4sig_cat.txt"))
            # print("rm -rf " + loc + cube_name + "_4sig_cat.txt")

        elif os.path.isfile(loc + cube_name + '_filtered_spline.fits'):
            print("\tProbably no sources cleaned for {}{}.fits? No filter-2d generated; deleting extra files".format(loc, cube_name))
            print("[FILECLEANUP] Deleting {}".format(loc + cube_name + ".fits"))
            os.system("rm -rf " + loc + cube_name + ".fits")
            print("[FILECLEANUP] Deleting {}".format(loc + cube_name + "_filtered.fits"))
            os.system("rm -rf " + loc + cube_name + "_filtered.fits")

        else:
            print("\tloc + cube_name")
            print("\t[WARNING]: How did I get here??????")

    print(
        "\n********************************************************************************************************\n")

    # if all_spline exists & filtered-2d does not exist then (means I went on to clean!)
        # Delete orig file;
        # Copy the mask-2d; copy nans from filtered file & save
            # Delete filtered (can remake from filter-2d & original file)
            # Delete filtered spline (can remake from filter-2d & all_spline)
        # Keep all-spline
        # Keep rep_clean & rep_clean_model
        # Delete beam
        # Keep cbcor
        # Delete cb

    # elif all_spline & filtered-2d do not exist then (means I didn't find any sources worth cleaning)
        # Delete orig file; delete 4sig_cat.txt; delete 4sig_mask*.fits; delete *4sig_chan.fits
        # Copy the mask-2d; copy nans from filtered file & save
            # Deleted filtered
            # Keep filtered spline
