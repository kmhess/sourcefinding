import os
import sys

from argparse import ArgumentParser, RawTextHelpFormatter
from astropy.io import ascii, fits
from astropy import units as u
import numpy as np

sys.path.insert(0, os.environ['SOFIA_MODULE_PATH'])
from sofia import writemoment2


def chan2freq(channels=None, hdu=None):
    frequencies = (channels * hdu[0].header['CDELT3'] + hdu[0].header['CRVAL3']) * u.Hz
    return frequencies


###################################################################

parser = ArgumentParser(description="Create new moment maps for (cleaned!) line cubes for a given taskid, beam, cubes",
                        formatter_class=RawTextHelpFormatter)

parser.add_argument('-t', '--taskid', default='190915041',
                    help='Specify the input taskid (default: %(default)s).')

parser.add_argument('-b', '--beams', default='0-39',
                    help='Specify a range (0-39) or list (3,5,7,11) of beams on which to do source finding (default: %(default)s).')

parser.add_argument('-c', '--cubes', default='1,2,3',
                    help='Specify the cubes on which to do source finding (default: %(default)s).')

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

HI_restfreq = 1420405751.77 * u.Hz
optical_HI = u.doppler_optical(HI_restfreq)

cube_name = 'HI_image_cube'
for b in beams:
    loc = '/tank/hess/apertif/' + taskid + '/B0' + str(b).zfill(2) + '/'

    for c in cubes:
        if os.path.isfile(loc + cube_name + '{}_clean.fits'.format(c)):
            # Make moment maps:
            cat = ascii.read(loc + cube_name + '{}_4sig_cat.txt'.format(c))
            print("Found {} sources in Beam {:02} Cube {}".format(len(cat), b, c))
            hdu_clean = fits.open(loc + cube_name + '{}_clean.fits'.format(c))
            hdu_mask3d = fits.open(loc + cube_name + '{}_4sig_mask.fits'.format(c))
            writemoment2.writeMoments(hdu_clean[0].data, hdu_mask3d[0].data, loc+cube_name+'{}_clean'.format(c),
                                      False, hdu_clean[0].header, False, True, True, True)

            bmaj = hdu_clean[0].header['BMAJ'] * 3600. * u.arcsec
            bmin = hdu_clean[0].header['BMIN'] * 3600. * u.arcsec
            cellsize = hdu_clean[0].header['CDELT2'] * 3600. * u.arcsec

            # Make HI profiles:  **** NEED TO IMPROVE THIS BY STARTING WITH 3D MASK?!! *****
            hdu_mask2d = fits.open(loc + cube_name + '{}_4sig_mask-2d.fits'.format(c))
            mask2d = hdu_mask2d[0].data[:, :]

            mask2d = np.asfarray(mask2d)
            mask2d[mask2d < 1] = np.nan

            cube_frequencies = chan2freq(np.array(range(hdu_clean[0].data.shape[0])), hdu=hdu_clean)

            for s in range(len(cat)):
                spectrum = np.sum(hdu_clean[0].data[:, mask2d == cat['col2'][s]], axis=1)
                ascii.write([cube_frequencies, spectrum], loc + 'HI_image_cube{}_clean_source{}.txt'.format(c, s),
                            names=['Frequency [Hz]', 'Flux [Jy/beam*pixel]'])
                os.system('echo "# BMAJ = {}\n# BMIN = {}\n# CELLSIZE = {:.2f}" > temp'.format(bmaj, bmin, cellsize))
                os.system('cat temp ' + loc + 'HI_image_cube{}_clean_source{}.txt'.format(c, s) +
                          ' > temp2 && mv temp2 ' + loc + 'HI_image_cube{}_clean_source{}.txt'.format(c, s))
                os.system('rm temp')

        else:
            print("No CLEAN cube for Beam {:02}, Cube {}".format(b, c))

print("Beam information can be read from the text files like: a.meta['comments'][0].replace('= ','').split()")
