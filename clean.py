import logging
import os

from argparse import ArgumentParser, RawTextHelpFormatter
from astropy.io import ascii
import astropy.io.fits as pyfits
import numpy as np

logger = logging.getLogger(__name__)

import apercal
from apercal.modules.base import BaseModule
from apercal.subs import setinit as subs_setinit
from apercal.subs import managefiles as subs_managefiles
from apercal.subs import imstats as subs_imstats
from apercal.libs import lib

from modules.functions import write_catalog


# class clean(BaseModule):
#     """
#     Clean class does HI source finding, cleaning, restoring within individual beams
#     """
#
#     cleandir = None
#
#     def __init__(self, file_=None, **kwargs):
#         self.default = lib.load_config(self, file_)
#         subs_setinit.setinitdirs(self)
#         subs_setinit.setdatasetnamestomiriad(self)
#
#     def go(self):
#         """
#
#         :return:
#         """
#         logger.info("Starting CLEANING ")
#         self.clean()
#         logger.info("CLEANING done ")
#
#     def sourcefinding(self):

# Retrieve cubes
# Check if cube exists
# Run SoFiA using some set of parameters that I like in sourcefinding.py  DONE (mostly).

# Check if there were sources detected
# Check the masks in checkmasks.py

# Clean the data iteratively...
# Return final cubelets??


###################################################################

parser = ArgumentParser(description="Do source finding in the HI spectral line cubes for a given taskid, beam, cubes",
                        formatter_class=RawTextHelpFormatter)

parser.add_argument('-t', '--taskid', default='190915041',
                    help='Specify the input taskid (default: %(default)s).')

parser.add_argument('-b', '--beams', default='0-39',
                    help='Specify a range (0-39) or list (3,5,7,11) of beams on which to do source finding (default: %(default)s).')

parser.add_argument('-c', '--cubes', default='1,2,3',
                    help='Specify the cubes on which to do source finding (default: %(default)s).')

parser.add_argument('-s', '--sources', default='all',
                    help='Specify sources to clean.  Can specify range or list. (default: %(default)s).')

parser.add_argument('-o', "--overwrite",
                    help="If option is included, overwrite old clean, model, and residual FITS files.",
                    action='store_true')

# Parse the arguments above
args = parser.parse_args()

###################################################################

# Range of cubes/beams to work on:
taskid = args.taskid
cubes = [int(c) for c in args.cubes.split(',')]

if '-' in args.beams:
    b_range = args.beams.split('-')
    beams = np.array(range(int(b_range[1])-int(b_range[0])+1)) + int(b_range[0])
else:
    beams = [int(b) for b in args.beams.split(',')]

if args.sources == 'all':
    mask_expr = '"<mask_sofia>.ge.0.5"'
elif '-' in args.sources:
    mask_range = args.sources.split('-')
    sources = np.array(range(int(mask_range[1])-int(mask_range[0])+1)) + int(mask_range[0])
    mask_expr = '"(<mask_sofia>.eq.0.5).or.((<mask_sofia>.ge.{}).and.(<mask_sofia>.le.{}))"'.format(mask_range[0], mask_range[1])
else:
    sources = [str(s) for s in args.sources.split(',')]
    mask_expr = '"(<mask_sofia>.eq.0.5).or.(<mask_sofia>.eq.'+').or.(<mask_sofia>.eq.'.join(sources)+')"'

overwrite = args.overwrite

cube_name = 'HI_image_cube'
beam_name = 'HI_beam_cube'
header = ['id', 'x', 'y', 'z', 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max', 'n_pix', 'f_min', 'f_max',
          'f_sum', 'rel', 'flag', 'taskid', 'beam', 'cube']

catParNames = ("id", "x", "y", "z", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max",
               "n_pix", "f_min", "f_max", "f_sum", "rel", "flag", "taskid", "beam", "cube")
catParUnits = ("-", "pix", "pix", "chan", "pix", "pix", "pix", "pix", "chan", "chan", "-", "JY/BEAM",
               "JY/BEAM", "JY/BEAM", "-", "-", "-", "-", "-")
catParFormt = ("%10i", "%10.3f", "%10.3f", "%10.3f", "%7i", "%7i", "%7i", "%7i", "%7i", "%7i", "%8i",
               "%10.7f", "%10.7f", "%12.6f", "%8.6f", "%7i", "%10i", "%7i", "%7i")

# prepare = apercal.prepare()

# for cube_counter in range(len(self.line_cube_channelwidth_list)):
for b in beams:
    loc = '/tank/hess/apertif/' + taskid + '/B0' + str(b).zfill(2) + '/'
    print(loc)
    clean_catalog = loc + 'clean_cat.txt'

    # subs_managefiles.director(self, 'ch', self.cleandir)
    # subs_managefiles.director(prepare, 'ch', loc)

    for c in cubes:
        # cube_name = self.cleandir + '/cubes/' + self.line_image_cube_name + '{0}.fits'.format(cube_counter)
        line_cube = loc + cube_name + '{0}.fits'.format(c)
        # beam_cube_name = self.cleandir + '/cubes/' + self.line_image_beam_cube_name + '{0}.fits'.format(cube_counter)
        beam_cube = loc + beam_name + '{0}.fits'.format(c)
        # mask_cube_name = self.cleandir + '/cubes/' + self.line_image_mask_cube_name+ '{0}.fits'.format(cube_counter)
        mask_cube = loc + cube_name + '{0}_4sig_mask.fits'.format(c)
        filter_cube = loc + cube_name + '{0}_filtered.fits'.format(c)
        catalog_file = loc + cube_name + '{0}_4sig_cat.txt'.format(c)

        if os.path.isfile(mask_cube):
            # Output what exactly is being used to clean the data
            print(mask_cube)
            # Edit mask cube to trick Miriad into using the whole volume.
            m = pyfits.open(mask_cube, mode='update')
            m[0].data[0, 0, 0] = -1
            m[0].data[-1, -1, -1] = -1
            m[0].scale('int16')
            m.flush()

            print("[CLEAN] Determining the statistics of Beam {:02}, Cube {}.".format(b, c))
            f = pyfits.open(filter_cube)
            mask = np.ones(f[0].data.shape[0], dtype=bool)
            if c == 3: mask[376:662] = False
            lineimagestats = [np.nanmin(f[0].data[mask]), np.nanmax(f[0].data[mask]), np.nanstd(f[0].data[mask])]
            f.close()
            print("[CLEAN] Image min, max, std: ", lineimagestats[:])

            if overwrite:
                os.system('rm -rf model_* beam_* map_* image_* mask_* residual_*')

            print("[CLEAN] Reading in FITS files, making Miriad mask.")

            fits = lib.miriad('fits')
            fits.op = 'xyin'
            fits.in_ = line_cube
            fits.out = 'map_00'
            fits.go()

            fits.in_ = beam_cube
            fits.out = 'beam_00'
            fits.go()

            fits.in_ = mask_cube
            fits.out = 'mask_sofia'
            fits.go()

            maths = lib.miriad('maths')
            maths.out = 'mask_00'
            maths.exp = '"<mask_sofia>"'
            maths.mask = mask_expr
            maths.go()

            nminiter = 1
            for minc in range(nminiter):
                print("[CLEAN] Cleaning HI emission using SoFiA mask for Sources {}.".format(args.sources))
                clean = lib.miriad('clean')
                clean.map = 'map_' + str(minc).zfill(2)
                clean.beam = 'beam_' + str(minc).zfill(2)
                clean.out = 'model_' + str(minc + 1).zfill(2)
                clean.cutoff = lineimagestats[2] * 0.5
                clean.region = '"' + 'mask(mask_' + str(minc).zfill(2) + '/)"'
                clean.go()

                print("[CLEAN] Restoring line cube.")
                restor = lib.miriad('restor')  # Create the restored image
                restor.model = 'model_' + str(minc + 1).zfill(2)
                restor.beam = 'beam_' + str(minc).zfill(2)
                restor.map = 'map_' + str(minc).zfill(2)
                restor.out = 'image_' + str(minc + 1).zfill(2)
                restor.mode = 'clean'
                restor.go()

                print("[CLEAN] Making residual cube.")
                restor.mode = 'residual'  # Create the residual image
                restor.out = 'residual_' + str(minc + 1).zfill(2)
                restor.go()

            if overwrite:
                os.system('rm {}_clean.fits {}_residual.fits {}_model.fits'.format(line_cube[:-5], line_cube[:-5], line_cube[:-5]))
                print("WARNING...overwrite won't delete clean_cat.txt file.  Manage this carefully!")

            print("[CLEAN] Writing out cleaned image, residual, and model to FITS.")
            fits.op = 'xyout'
            fits.in_ = 'image_' + str(minc + 1).zfill(2)
            fits.out = line_cube[:-5] + '_clean.fits'
            fits.go()

            fits.in_ = 'residual_' + str(minc + 1).zfill(2)
            fits.out = line_cube[:-5] + '_residual.fits'
            fits.go()

            fits.in_ = 'model_' + str(minc + 1).zfill(2)
            fits.out = line_cube[:-5] + '_model.fits'
            fits.go()

            # If everything was successful and didn't crash for a given beam/cube:
            catalog = ascii.read(catalog_file, header_start=10)
            catalog['taskid'] = np.int(taskid.replace('/',''))
            catalog['beam'] = b
            catalog['cube'] = c
            catalog_reorder = catalog['id', 'x', 'y', 'z',  'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max',
                                      'n_pix', 'f_min', 'f_max', 'f_sum', 'rel', 'flag', 'taskid', 'beam', 'cube']
            objects = []
            for source in catalog_reorder:
                obj = []
                for s in source:
                    obj.append(s)
                objects.append(obj)

            write_catalog(objects, catParNames, catParUnits, catParFormt, header, outName=loc+'clean_cat.txt')

            # Clean up extra Miriad files
            os.system('rm -rf model_* beam_* map_* image_* mask_* residual_*')

    # Will probably need to do some sorting of the catalog if run clean multiple times.  This is a starting point:
    # os.system('head -n +1 {} > temp'.format(clean_catalog))
    # os.system('tail -n +2 {} | sort | uniq > temp2'.format(clean_catalog))

print("[CLEAN] Done.")
