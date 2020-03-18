import logging
import os

import astropy.io.fits as pyfits
import numpy as np

# logger = logging.getLogger(__name__)

import apercal
from apercal.modules.base import BaseModule
from apercal.subs import setinit as subs_setinit
from apercal.subs import managefiles as subs_managefiles
from apercal.subs import imstats as subs_imstats
from apercal.libs import lib


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


# Range of cubes/beams to work on:
taskid = '190914041'
cubes = [1, 2, 3]  # Most sources in 2; nearest galaxies in 3.
beams = range(40)

cube_name = 'HI_image_cube'
beam_name = 'HI_beam_cube'

prepare = apercal.prepare()
# subs_managefiles.director(self, 'ch', self.cleandir)
subs_managefiles.director(prepare, 'ch', loc)

# for cube_counter in range(len(self.line_cube_channelwidth_list)):
for c in cubes:
    for b in beams:
        loc = '/tank/hess/apertif/' + taskid + '/B0' + str(beams).zfill(2) + '/'

        # cube_name = self.cleandir + '/cubes/' + self.line_image_cube_name + '{0}.fits'.format(cube_counter)
        line_cube = loc + cube_name + '{0}.fits'.format(c)
        # beam_cube_name = self.cleandir + '/cubes/' + self.line_image_beam_cube_name + '{0}.fits'.format(cube_counter)
        beam_cube = loc + beam_name + '{0}.fits'.format(c)
        # mask_cube_name = self.cleandir + '/cubes/' + self.line_image_mask_cube_name+ '{0}.fits'.format(cube_counter)
        mask_cube = loc + cube_name + '{0}_4sig_mask.fits'.format(c)
        filter_cube = loc + cube_name + '{0}_filtered_sof1.fits'.format(c)

        if os.path.isfile(mask_cube):
            print("[CLEAN] Determining the statistics of Cube {}, beam {:02}.".format(c, b))
            image_data = pyfits.open(filter_cube)
            data = image_data[0].data
            lineimagestats = np.full(3, np.nan)
            if data.shape[-3] == 2:
                lineimagestats[0] = np.nanmin(data[0, 0, :, :])  # Get the maxmimum of the image
                lineimagestats[1] = np.nanmax(data[0, 0, :, :])  # Get the minimum of the image
                lineimagestats[2] = np.nanstd(data[0, 0, :, :])  # Get the standard deviation
            else:
                lineimagestats[0] = np.nanmin(data)  # Get the maxmimum of the image
                lineimagestats[1] = np.nanmax(data)  # Get the minimum of the image
                lineimagestats[2] = np.nanstd(data)  # Get the standard deviation
            image_data.close()
            print("[CLEAN] Image min, max, std: ", lineimagestats[:])

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
            maths.mask = '"<mask_sofia>.eq.1"'
            maths.go()

            nminiter = 1
            for minc in range(nminiter):
                print("[CLEAN] Cleaning HI emission using SoFiA mask.")
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

print("[CLEAN] Done.")
