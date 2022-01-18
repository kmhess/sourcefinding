#!/home/apercal/pipeline/bin/python3
# change the line above to reflect the location of your local python3 executable
#
__author__ = "Thijs van der Hulst"  # Adapted here by Kelley M. Hess
__email__ = "vdhulst@astro.rug.nl"
#
# runs under the Apercal environment at ASTRON
# useage: <path_to>/trim_beam.py <inputbeam> <outputbeam> <-1> for halving a beam cube
# and:    <path_to>/trim_beam.py <inputbeam> <outputbeam> <+1> for recreating a full beam cube
#
# alternatively one could run the script as:
#   python3 <path_to>/trim_beam.py <inputbeam> <outputbeam> <nr>
#
import numpy as np
import os
import sys
import math
from astropy.io import fits

##########################################################################################


def main(filein, fileout, direction):
    if not os.path.isfile (filein):
        sys.stderr.write ("specified input beam does not exist \n");
        sys.exit (1);

    if os.path.isfile (fileout):
        sys.stderr.write ("specified output beam exists, please specify another name \n");
        sys.exit (1);

    fitsfile = fits.open (filein, memmap=True, do_not_scale_image_data=True)

    if direction == -1:

        # Open the file to get the header information and array sizes
        # Apertif line cubes are 1321 x 1321 x 1218 pixels, reduce to
        # 661 x 1321 x 1218 pixels

        fitsheader = fitsfile[0].header
        naxis = fitsheader['NAXIS']
        naxis1 = fitsheader['NAXIS1']
        naxis2 = fitsheader['NAXIS2']
        naxis3 = fitsheader['NAXIS3']
        l_end = int ((naxis1 + 1) / 2)

        cube = np.full ((naxis3, naxis2, l_end), np.nan, dtype='float32')
        fitsfile_data = fitsfile[0].data
        fitsheader['NAXIS1'] = (l_end, " ")
        cube = fitsfile_data[0:naxis3, 0:naxis2, 0:l_end]

        fits.writeto (fileout, cube, fitsheader, overwrite=False)
        fitsfile.close ()

    else:
        if direction == 1:
            # Open the file to get the header information and array sizes
            # Apertif half beam cubes are 661 x 1321 x 1218 pixels, expand to
            # 1321 x 1321 x 1218 pixels

            fitsheader = fitsfile[0].header
            naxis = fitsheader['NAXIS']
            naxis1 = fitsheader['NAXIS1']
            naxis2 = fitsheader['NAXIS2']
            naxis3 = fitsheader['NAXIS3']
            l_end = int (2 * naxis1 - 1)

            # expand the cube to its full size by adding a flipped (in x and y) version of itself

            fitsheader['NAXIS1'] = (l_end, " ")
            cube1 = fitsfile[0].data
            cube2 = np.flip (cube1[:, :, 0:naxis1 - 1], axis=1)
            cube2 = np.flip (cube2[:, :, 0:naxis1 - 1], axis=2)
            cube = np.concatenate ((cube1, cube2), axis=2)

            fits.writeto (fileout, cube, fitsheader, overwrite=False)
            fitsfile.close ()

if __name__ == '__main__':

    if len (sys.argv) < 4:
        sys.stderr.write ("Usage: ./trim_beam.py <inputbeam> <outputbeam> <-1 or +1>\n");
        sys.exit (1);

    filein = sys.argv[1]
    fileout = sys.argv[2]
    direction = int (sys.argv[3])

    main(filein, fileout, direction)
