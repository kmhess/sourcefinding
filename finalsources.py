import os
import sys

from argparse import ArgumentParser, RawTextHelpFormatter
from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits
from astropy import units as u
from astropy.wcs import WCS
from astroquery.skyview import SkyView
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from reproject import reproject_interp

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
            # Read in the master catalog of cleaned sources
            catalog = ascii.read(loc + 'clean_cat.txt')
            cat = catalog[catalog['cube'] == c]
            cathead = np.array(cat.colnames)
            print("Found {} sources in Beam {:02} Cube {}".format(len(cat), b, c))

            # Read in the cleaned data and original SoFiA mask:
            hdu_clean = fits.open(loc + cube_name + '{}_clean.fits'.format(c))
            hdu_mask3d = fits.open(loc + cube_name + '{}_4sig_mask.fits'.format(c))
            outname = 'src_taskid{}_beam{:02}_cube{}'.format(taskid, b, c)
            wcs = WCS(hdu_clean[0].header)

            # Make cubelets around each individual source, mom0,1,2 maps, and sub-spectra from cleaned data
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
            bpa = hdu_clean[0].header['BPA']
            hi_cellsize = hdu_clean[0].header['CDELT2'] * 3600. * u.arcsec
            opt_view = 6. * u.arcmin
            opt_pixels = 900

            # Make HI profiles with noise over whole cube by squashing 3D mask:
            cube_frequencies = chan2freq(np.array(range(hdu_clean[0].data.shape[0])), hdu=hdu_clean)
            for obj in objects:
                # Some lines stolen from cubelets in  SoFiA:
                cubeDim = hdu_clean[0].data.shape
                Xc = obj[cathead == "x"][0]
                Yc = obj[cathead == "y"][0]
                Xmin = obj[cathead == "x_min"][0]
                Ymin = obj[cathead == "y_min"][0]
                Xmax = obj[cathead == "x_max"][0]
                Ymax = obj[cathead == "y_max"][0]
                cPixXNew = int(Xc)
                cPixYNew = int(Yc)
                maxX = 2 * max(abs(cPixXNew - Xmin), abs(cPixXNew - Xmax))
                maxY = 2 * max(abs(cPixYNew - Ymin), abs(cPixYNew - Ymax))
                XminNew = cPixXNew - maxX
                if XminNew < 0: XminNew = 0
                YminNew = cPixYNew - maxY
                if YminNew < 0: YminNew = 0
                XmaxNew = cPixXNew + maxX
                if XmaxNew > cubeDim[2] - 1: XmaxNew = cubeDim[2] - 1
                YmaxNew = cPixYNew + maxY
                if YmaxNew > cubeDim[1] - 1: YmaxNew = cubeDim[1] - 1

                # Array math a lot faster on (spatially) tiny subcubes:
                subcube = hdu_clean[0].data[:, int(YminNew):int(YmaxNew) + 1, int(XminNew):int(XmaxNew) + 1]
                submask = fits.getdata(loc + outname + '_{}_mask.fits'.format(int(obj[0])))

                # Can potentially save mask2d as a better nchan if need be because mask values are 0 or 1:
                mask2d = np.sum(submask, axis=0)
                spectrum = np.nansum(subcube[:, mask2d != 0], axis=1)
                ascii.write([cube_frequencies, spectrum], loc + outname + '_{}_specfull.txt'.format(int(obj[0])),
                            names=['Frequency [Hz]', 'Flux [Jy/beam*pixel]'])
                os.system('echo "# BMAJ = {}\n# BMIN = {}\n# CELLSIZE = {:.2f}" > temp'.format(bmaj, bmin, hi_cellsize))
                os.system('cat temp ' + loc + outname + '_{}_specfull.txt'.format(int(obj[0])) + ' > temp2 && mv temp2 '
                          + loc + outname + '_{}_specfull.txt'.format(int(obj[0])))
                os.system('rm temp')

                # Get optical image
                subcoords = wcs.wcs_pix2world(Xc, Yc, 1, 1)  # .to_string('hmsdms')
                c = SkyCoord(ra=subcoords[0], dec=subcoords[1], unit=u.deg)
                path = SkyView.get_images(position=c.to_string('hmsdms'), width=opt_view, height=opt_view,
                                          survey=['DSS2 Blue'], pixels=[opt_pixels, opt_pixels])
                name = c.to_string('hmsdms')

                if (len(path) != 0) & (not os.path.isfile(loc + outname + '_{}_overlay.png'.format(int(obj[0])))):
                    # Get optical image and HI subimage
                    hdulist_opt = path[0]
                    d2 = hdulist_opt[0].data
                    h2 = hdulist_opt[0].header
                    hdulist_hi = fits.open(loc + outname + '_{}_mom0.fits'.format(int(obj[0])))

                    # Reproject HI data & calculate contour properties
                    hi_reprojected, footprint = reproject_interp(hdulist_hi, h2)
                    chan_width = hdu_clean[0].header['CDELT3']
                    rms = np.std(subcube) * chan_width
                    nhi19 = 2.33e20 * rms / (bmaj.value * bmin.value) / 1e19
                    print("NHI is {}e+19".format(nhi19))
                    nhi_label = "N_HI ={:4.1f}, {:4.1f}, {:4.1f}, {:4.1f}, {:4.1f}, {:4.1f}e+19".format(nhi19 * 3, nhi19 * 5,
                                                                                                        nhi19 * 10, nhi19 * 20,
                                                                                                        nhi19 * 40, nhi19 * 80)
                    # Overlay HI contours on optical image
                    fig = plt.figure(figsize=(8, 8))
                    ax1 = fig.add_subplot(111, projection=WCS(hdulist_opt[0].header))
                    ax1.imshow(d2, cmap='viridis', vmin=np.percentile(d2, 10), vmax=np.percentile(d2, 99.8))
                    ax1.contour(hi_reprojected, cmap='Oranges', levels=[rms * 3, rms * 5, rms * 10, rms * 20, rms * 40, rms * 80])
                    ax1.set_title(name, fontsize=20)
                    ax1.tick_params(axis='both', which='major', labelsize=18)
                    ax1.coords['ra'].set_axislabel('RA (J2000)', fontsize=20)
                    ax1.coords['dec'].set_axislabel('Dec (J2000)', fontsize=20)
                    ax1.text(0.5, 0.05, nhi_label, ha='center', va='center', transform=ax1.transAxes, color='white', fontsize=18)
                    ax1.add_patch(Ellipse((0.92, 0.9), height=(bmaj/opt_view).decompose(), width=(bmin/opt_view).decompose(),
                                          angle=bpa, transform=ax1.transAxes, edgecolor='white', linewidth=1))

                    fig.savefig(loc + outname + '_{}_overlay.png'.format(int(obj[0])), bbox_inches='tight')
                    hdulist_hi.close()

            hdu_clean.close()
            hdu_mask3d.close()

        else:
            print("No CLEAN cube for Beam {:02}, Cube {}".format(b, c))

print("Beam information in *specfull.txt can be read from the text files like: a.meta['comments'][0].replace('= ','').split()")
print("[FINALSOURCES] Done.")
