#!/usr/bin/env python

# This script does primary beam correction for apertif HI cubes, using the beam models derived from the drift scans
# At the moment the bin 9 I beam models are used for the PB correction, this model is the closest in frequency to cube2.  
# PB models on happili-05 are in: /tank/apertif/driftscans/fits_files/191023/beam_models/
# for cube2 I used chann_9 beam models, these are closest in frequency
# steps: 
# 1, change centre for the beam model
# 2, regrid beam image to the same size as the HI cube in RA and DEC (using miriad for this)
# 3, make a frequency axes for the beam map, devide the Image cube with the beam cube (PB correction) 

import os
import numpy as np
from astropy.io import fits as pyfits
from apercal.libs import lib


# ------------------------------------------------
def regrid_in_miriad(image_name, cb2d_name, hdu_image, b, c):
	"""
	Find appropriate beam model and set center to center of image.
	Regrid the beam model image in miriad to the HI image.
	Expand beam model imagine in 3D.
	"""

	cb_model_dir = '/tank/apertif/driftscans/fits_files/191023/beam_models/chann_9/'

	hdulist_cb = pyfits.open(cb_model_dir + '191023_{:02}_I_model.fits'.format(b))
	hdulist_cb[0].header['CRVAL1'] = hdu_image[0].header['CRVAL1']
	hdulist_cb[0].header['CRVAL2'] = hdu_image[0].header['CRVAL2']
	hdulist_cb.writeto(cb2d_name)
	hdulist_cb.close()

	print('Regridding in miriad')
	
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

	print('Writing beam cube.')
	hdu_cb.writeto('{}_cb.fits'.format(image_name[:-5]))

	hdu_cb.close()

	# Clean up the extra Miriad & 2D cb files
	os.system('rm -rf *.mir')
	os.system('rm -rf {}*'.format(cb2d_name[:-5]))


# ----------------------------------------------
def apply_pb(hdu_image, hdu_cb, image_name):
	"""
	Apply the beam cube to do the compound beam correction.
	chan_range produces a smaller HI cube with a subset of channels.
	Writes out the beam cube as a fits file and the PB corrected cube.
	"""
	print('Doing compound beam correction')

	cbcor = pyfits.PrimaryHDU(hdu_image[0].data / hdu_cb[0].data, header=hdu_image[0].header)
	cbcor.writeto('{}_cbcor.fits'.format(image_name[:-5]))

	return cbcor
