import os
import sys

from argparse import ArgumentParser, RawTextHelpFormatter
from astropy.io import ascii, fits
from astropy import units as u
import numpy as np

sys.path.insert(0, os.environ['SOFIA_MODULE_PATH'])
from sofia import cubelets

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

cube_name = 'HI_image_cube'
for b in beams:
    loc = '/tank/hess/apertif/' + taskid + '/B0' + str(b).zfill(2) + '/'

    for c in cubes:
        if os.path.isfile(loc + cube_name + '{}_clean.fits'.format(c)):
            catalog = ascii.read(loc + 'clean_cat.txt')
            cat = catalog[catalog['cube'] == c]
            cathead = np.array(cat.colnames)
            print("Found {} sources in Beam {:02} Cube {}".format(len(cat), b, c))

            # Make moment maps:
            hdu_clean = fits.open(loc + cube_name + '{}_clean.fits'.format(c))
            hdu_mask3d = fits.open(loc + cube_name + '{}_4sig_mask.fits'.format(c))
            outname = 'src_taskid{}_beam{:02}_cube{}'.format(taskid, b, c)

            # Make cubelets around each individual source, mom0,1,2 maps, and sub-spectra
            objects = []
            for source in cat:
                obj = []
                for s in source:
                    obj.append(s)
                objects.append(obj)
            objects = np.array(objects)
            cubelets.writeSubcube(hdu_clean[0].data, hdu_clean[0].header, hdu_mask3d[0].data, objects, cathead,
                                  outname, loc, False, False)

            # Get beam size and cell size
            bmaj = hdu_clean[0].header['BMAJ'] * 3600. * u.arcsec
            bmin = hdu_clean[0].header['BMIN'] * 3600. * u.arcsec
            cellsize = hdu_clean[0].header['CDELT2'] * 3600. * u.arcsec

            # Make individual HI profiles over whole cube by squashing 3D mask:
            cube_frequencies = chan2freq(np.array(range(hdu_clean[0].data.shape[0])), hdu=hdu_clean)
            for obj in objects:
                # Array math hopefully a lot faster on (spatially) tiny subcubes...???
                subcube = hdu_clean[0].data[:, int(obj[cathead == 'y_min'][0]):int(obj[cathead == 'y_max'][0] + 1),
                          int(obj[cathead == 'x_min'][0]):int(obj[cathead == 'x_max'][0] + 1)]
                submask = hdu_mask3d[0].data[:, int(obj[cathead == 'y_min'][0]):int(obj[cathead == 'y_max'][0] + 1),
                          int(obj[cathead == 'x_min'][0]):int(obj[cathead == 'x_max'][0] + 1)]

                mask_one = np.zeros(subcube.shape)
                mask_one[submask == obj[0]] = 1
                # Can potentially save this as a better nchan if need be:
                mask2d = np.sum(mask_one, axis=0)
                spectrum = np.nansum(subcube[:, mask2d != 0], axis=1)
                ascii.write([cube_frequencies, spectrum], loc + outname + '_{}_specfull.txt'.format(int(obj[0])),
                            names=['Frequency [Hz]', 'Flux [Jy/beam*pixel]'])
                os.system('echo "# BMAJ = {}\n# BMIN = {}\n# CELLSIZE = {:.2f}" > temp'.format(bmaj, bmin, cellsize))
                os.system('cat temp ' + loc + outname + '_{}_specfull.txt'.format(int(obj[0])) + ' > temp2 && mv temp2 '
                          + loc + outname + '_{}_specfull.txt'.format(int(obj[0])))
                os.system('rm temp')

            hdu_clean.close()
            hdu_mask3d.close()

        else:
            print("No CLEAN cube for Beam {:02}, Cube {}".format(b, c))

print("Beam information in *specfull.txt can be read from the text files like: a.meta['comments'][0].replace('= ','').split()")
print("[FINALSOURCES] Done.")