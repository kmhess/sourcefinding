#!/usr/bin/env python

# This script does primary beam correction for apertif HI cubes, using the beam models derived from the drift scans
# At the moment the bin 9 I beam models are used for the PB correction, this model is the closest in frequency to cube2.  
# PB models on happili-05 are in: /tank/apertif/driftscans/fits_files/191023/beam_models/
# for cube2 I used chann_9 beam models, these are closest in frequency
# steps: 
# 1, change centre for the beam model
# 2, regrid beam image to the same size as the HI cube in RA and DEC (using miriad for this)
# 3, make a frequency axes for the beam map, devide the Image cube with the beam cube (PB correction) 

# Script formerly known as PB_correction_happili2.py

import os

from astropy.io import fits as pyfits
import astropy.units as u
import numpy as np
from apercal.libs import lib

from modules import beam_lookup


# ----------------------------------------------
def pbcor(taskid, image_name, hdu_image, beam, cube):
	"""
	Find and regrid the model beam to match the image.
	Apply primary beam correction.
	:param image_name:
	:param hdu_image:
	:param beam:
	:param cube:
	:param chan_range:
	:return:
	"""

	# Make regridded CB FITS file if it doesn't already exist:
	if not os.path.isfile('{}_cb.fits'.format(image_name[:-5])) | \
		   os.path.isfile('{}_cbcor.fits'.format(image_name[:-5])):
		regrid_in_miriad(taskid, image_name, hdu_image, beam, cube)

	# Make cbcor'ed FITS file if it doesn't already exist:
	if not os.path.isfile('{}_cbcor.fits'.format(image_name[:-5])):
		hdu_cb = pyfits.open('{}_cb.fits'.format(image_name[:-5]))
		apply_pb(hdu_image, hdu_cb, image_name)
		hdu_cb.close()
	else:
		print("\tCompound beam corrected image exists.  Load existing image.")

	return


# ------------------------------------------------
def get_cb_model_freq():
	"""
	Set the central frequency for the Gaussian regression beams based on the Apertif DR1 documentation.
	"""
	alexander_orig_dr1 = 1361.25 * u.MHz
	return alexander_orig_dr1


def regrid_in_miriad(taskid, image_name, hdu_image, b, c):
	"""
	Find appropriate beam model and set center to center of image.
	Rescale the beam model to appropriate size for the center frequency of the cube.
	Regrid the beam model image in miriad to the HI image.
	Expand beam model imagine in 3D.
	"""

	# Change the reference pixel of beam model to reference pixel of image to correct
	cb_model = beam_lookup.model_lookup2(taskid, b)
	hdulist_cb = pyfits.open(cb_model)
	hdulist_cb[0].header['CRVAL1'] = hdu_image[0].header['CRVAL1']
	hdulist_cb[0].header['CRVAL2'] = hdu_image[0].header['CRVAL2']

	# Rescale to appropriate frequency. This should work for either drift scans or Gaussian regression (only tested on latter):
	avg_cube_freq = (hdu_image[0].header['CRVAL3'] + hdu_image[0].header['CDELT3'] * hdu_image[0].data.shape[0]) * u.Hz
	hdulist_cb[0].header['CDELT1'] = (hdulist_cb[0].header['CDELT1'] * get_cb_model_freq().to(u.Hz) / avg_cube_freq).value
	hdulist_cb[0].header['CDELT2'] = (hdulist_cb[0].header['CDELT2'] * get_cb_model_freq().to(u.Hz) / avg_cube_freq).value

	cb2d_name = 'temp_b{}_c{}_cb-2d.fits'.format(b, c)
	hdulist_cb.writeto(cb2d_name)
	hdulist_cb.close()

	print('\tRegridding in miriad using model {}'.format(cb_model))

	fits = lib.miriad('fits')
	regrid = lib.miriad('regrid')

	# Convert files to miriad:
	fits.in_ = image_name
	fits.out = '{}.mir'.format(image_name[:-5])
	fits.op = 'xyin'
	fits.go()

	fits.in_ = cb2d_name
	fits.out = '{}.mir'.format(cb2d_name[:-5])
	fits.op = 'xyin'
	fits.go()

	# Regrid beam image
	regrid.in_ = '{}.mir'.format(cb2d_name[:-5])
	regrid.out = '{}_rgrid.mir'.format(cb2d_name[:-5])
	regrid.tin = '{}.mir'.format(image_name[:-5])
	regrid.axes = '1,2'
	regrid.go()

	# Convert regrided beam image to fits
	fits.in_ = '{}_rgrid.mir'.format(cb2d_name[:-5])
	fits.out = '{}_rgrid.fits'.format(cb2d_name[:-5])
	fits.op = 'xyout'
	fits.go()

	# Make cb 3D and save as FITS:
	hdu_cb = pyfits.open('{}_rgrid.fits'.format(cb2d_name[:-5]))
	d_new = np.ones((hdu_image[0].header['NAXIS3'], hdu_cb[0].header['NAXIS2'], hdu_cb[0].header['NAXIS2']))
	d_beam_cube = d_new * hdu_cb[0].data
	hdu_cb[0].data = np.float32(d_beam_cube)

	print('\tWriting compound beam cube.')
	hdu_cb.writeto('{}_cb.fits'.format(image_name[:-5]))

	hdu_cb.close()

	# Clean up the extra Miriad & 2D cb files
	os.system('rm -rf {}*.mir'.format(image_name[:-5]))
	os.system('rm -rf {}*'.format(cb2d_name[:-5]))


# ----------------------------------------------
def apply_pb(hdu_image, hdu_cb, image_name):
	"""
	Apply the beam cube to do the compound beam correction.
	chan_range produces a smaller HI cube with a subset of channels.
	Writes out the beam cube as a fits file and the PB corrected cube.
	"""
	print('[APPLY_PB] Doing compound beam correction.')

	cbcor = pyfits.PrimaryHDU(hdu_image[0].data / hdu_cb[0].data, header=hdu_image[0].header)
	cbcor.writeto('{}_cbcor.fits'.format(image_name[:-5]))

	return
