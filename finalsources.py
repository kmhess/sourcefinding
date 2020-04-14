import os
import sys

from argparse import ArgumentParser, RawTextHelpFormatter
from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits
from astropy import units as u
from astropy.wcs import WCS
from astroquery.skyview import SkyView
from cosmocalc import cosmocalc
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from reproject import reproject_interp

sys.path.insert(0, os.environ['SOFIA_MODULE_PATH'])
from sofia import cubelets

from modules.functions import pbcor
from modules.functions import write_catalog

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
H0 = 70.

cube_name = 'HI_image_cube'

header = ['name', 'id', 'x', 'y', 'z', 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max',
          'logMhi', 'SJyHz', 'redshift', 'v_sys', 'D_Lum', 'rms_spec', 'SNR',
          'flag', 'rms', 'w20', 'w50', 'kin_pa', 'taskid', 'beam', 'cube']

catParNames = ("name", "id", "x", "y", "z", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "n_pix",
               "f_min", "f_max", "f_sum", "rel", "flag", "rms", "w20", "w50", "ell_maj", "ell_min", "ell_pa",
               "ell3s_maj", "ell3s_min", "ell3s_pa", "kin_pa", "taskid", "beam", "cube",
               "SJyHz", "logMhi", "redshift", "v_sys", "D_Lum", "rms_spec", "SNR")
catParUnits = ("-", "-", "pix", "pix", "chan", "pix", "pix", "pix", "pix", "chan", "chan", "-",
               "Jy/beam", "Jy/beam", "Jy/beam", "-", "-", "dunits", "chan", "chan", "pix", "pix", "pix",
               "pix", "pix", "deg", "deg", "-", "-", "-",
               "Jy*Hz", "log(M_Sun)", "-", "km/s", "Mpc", "Jy/chan", "-")
catParFormt = ("%15s", "%10i", "%10.3f", "%10.3f", "%10.3f", "%7i", "%7i", "%7i", "%7i", "%7i", "%7i", "%8i",
               "%10.7f", "%10.7f", "%12.6f", "%8.6f", "%7i", "%12.6f", "%10.3f", "%10.3f", "%10.3f", "%10.3f", "%10.3f",
               "%10.3f", "%10.3f", "%10.3f", "%10.3f", "%10i", "%7i", "%7i",
               "%13.6f", "%12.6f", "%11.7f", "%11.3f", "%10.3f", "%11.7f", "%8.3f")

for b in beams:
    loc = '/tank/hess/apertif/' + taskid + '/B0' + str(b).zfill(2) + '/'
    if os.path.isfile(loc + 'clean_cat.txt'):
        # Read in the master catalog of cleaned sources
        catalog = ascii.read(loc + 'clean_cat.txt', header_start=1)

        for c in cubes:
            if os.path.isfile(loc + cube_name + '{}_clean.fits'.format(c)):
                cat = catalog[catalog['cube'] == c]
                cathead = np.array(cat.colnames)[1:]    # This is to avoid issues with the name column in writeSubcube.
                print("Found {} sources in Beam {:02} Cube {}".format(len(cat), b, c))

                # Read in the cleaned data and original SoFiA mask:
                hdu_clean = fits.open(loc + cube_name + '{}_clean.fits'.format(c))
                hdu_mask3d = fits.open(loc + cube_name + '{}_4sig_mask.fits'.format(c))

                hdu_pb = pbcor(loc + cube_name + '{}_clean.fits'.format(c),
                               loc + cube_name + '{}_cb-2d.fits'.format(c), hdu_clean, b, c)

                outname = 'src_taskid{}_beam{:02}_cube{}new'.format(taskid, b, c)
                wcs = WCS(hdu_clean[0].header)

                # Make cubelets around each individual source, mom0,1,2 maps, and sub-spectra from cleaned data
                # Note that the name column causes big issues because it forces np.array to cast to string, not float!
                objects = []
                for source in cat:
                    obj = []
                    for s in source:
                        obj.append(s)
                    objects.append(obj[1:])
                objects = np.array(objects)
                cubelets.writeSubcube(hdu_pb[0].data, hdu_clean[0].header, hdu_mask3d[0].data, objects, cathead,
                                      outname, loc, False, False)

                # Get beam size and cell size
                bmaj = hdu_clean[0].header['BMAJ'] * 3600. * u.arcsec
                bmin = hdu_clean[0].header['BMIN'] * 3600. * u.arcsec
                bpa = hdu_clean[0].header['BPA']
                hi_cellsize = hdu_clean[0].header['CDELT2'] * 3600. * u.arcsec
                pix_per_beam = bmaj/hi_cellsize * bmin/hi_cellsize * np.pi / (4 * np.log(2))
                chan_width = hdu_clean[0].header['CDELT3']
                opt_view = 6. * u.arcmin
                opt_pixels = 900

                # Make HI profiles with noise over whole cube by squashing 3D mask:
                cube_frequencies = chan2freq(np.array(range(hdu_clean[0].data.shape[0])), hdu=hdu_clean)
                SJyHz, logMhi, redshift, v_sys, D_Lum, rms_spec, SNR = [], [], [], [], [], [], []
                for obj in objects:
                    # Some lines stolen from cubelets in  SoFiA:
                    cubeDim = hdu_clean[0].data.shape
                    Xc = obj[cathead == "x"][0]
                    Yc = obj[cathead == "y"][0]
                    Zc = obj[cathead == "z"][0]
                    Xmin = obj[cathead == "x_min"][0]
                    Ymin = obj[cathead == "y_min"][0]
                    Zmin = obj[cathead == "z_min"][0]
                    Xmax = obj[cathead == "x_max"][0]
                    Ymax = obj[cathead == "y_max"][0]
                    Zmax = obj[cathead == "z_max"][0]
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

                    # Do some prep for mom1 maps:
                    freqmin = chan2freq(Zmin,hdu_pb)
                    freqmax = chan2freq(Zmax,hdu_pb)
                    velmax = freqmin.to(u.km/u.s, equivalencies=optical_HI).value
                    velmin = freqmax.to(u.km/u.s, equivalencies=optical_HI).value

                    # Array math a lot faster on (spatially) tiny subcubes from cubelets.writeSubcubes:
                    subcube = hdu_pb[0].data[:, int(YminNew):int(YmaxNew) + 1, int(XminNew):int(XmaxNew) + 1]
                    submask = fits.getdata(loc + outname + '_{}_mask.fits'.format(int(obj[0])))
                    # Can potentially save mask2d as a better nchan if need be because mask values are 0 or 1:
                    mask2d = np.sum(submask, axis=0)

                    # Calculate spectrum and some fundamental galaxy parameters
                    spectrum = np.nansum(subcube[:, mask2d != 0], axis=1)
                    signal = np.nansum(spectrum[int(Zmin):int(Zmax)])
                    SJyHz.append(signal * chan_width / pix_per_beam)
                    freq_sys = chan2freq(channels=int(obj[cathead == "z"][0]), hdu=hdu_clean)
                    v_sys.append(freq_sys.to(u.km/u.s, equivalencies=optical_HI).value)
                    redshift.append(HI_restfreq / freq_sys - 1.)
                    cosmo = cosmocalc(redshift[-1], H0)
                    D_Lum.append(cosmo['DL_Mpc'])
                    Mhi = 49.7 * SJyHz[-1] * cosmo['DL_Mpc']**2
                    logMhi.append(np.log10(Mhi))
                    specmask = np.zeros(len(spectrum))
                    specmask[int(Zmin):int(Zmax)] = 1
                    rms_spec.append(np.std(spectrum[specmask == 0]))
                    SNR.append(signal / (rms_spec[-1] * np.sqrt(Zmax - Zmin)))

                    # Save spectrum to a txt file:
                    ascii.write([cube_frequencies, spectrum], loc + outname + '_{}_specfull.txt'.format(int(obj[0])),
                                names=['Frequency [Hz]', 'Flux [Jy/beam*pixel]'])
                    os.system('echo "# BMAJ = {}\n# BMIN = {}\n# CELLSIZE = {:.2f}" > temp'.format(bmaj, bmin,
                                                                                                   hi_cellsize))
                    os.system('cat temp ' + loc + outname + '_{}_specfull.txt'.format(int(obj[0])) +
                              ' > temp2 && mv temp2 ' + loc + outname + '_{}_specfull.txt'.format(int(obj[0])))
                    os.system('rm temp')

                    # Get optical image
                    subcoords = wcs.wcs_pix2world(Xc, Yc, 1, 1)  # .to_string('hmsdms')
                    c = SkyCoord(ra=subcoords[0], dec=subcoords[1], unit=u.deg)
                    path = SkyView.get_images(position=c.to_string('hmsdms'), width=opt_view, height=opt_view,
                                              survey=['DSS2 Blue'], pixels=[opt_pixels, opt_pixels])
                    name = c.to_string('hmsdms')
                    # name = 'AHC J{0}{1}'.format(c.ra.to_string(unit=u.hourangle, sep='', precision=1, pad=True),
                    #                             c.dec.to_string(sep='', precision=0, alwayssign=True, pad=True))

                    if (len(path) != 0):
                        # Get optical image and HI subimage
                        hdulist_opt = path[0]
                        d2 = hdulist_opt[0].data
                        h2 = hdulist_opt[0].header

                        if not os.path.isfile(loc + outname + '_{}_overlay.png'.format(int(obj[0]))):
                            hdulist_hi = fits.open(loc + outname + '_{}_mom0.fits'.format(int(obj[0])))
                            # Reproject HI data & calculate contour properties
                            hi_reprojected, footprint = reproject_interp(hdulist_hi, h2)
                            rms = np.std(subcube) * chan_width
                            nhi19 = 2.33e20 * rms / (bmaj.value * bmin.value) / 1e19
                            print("1sig N_HI is {}e+19".format(nhi19))
                            nhi_label = "N_HI ={:4.1f}, {:4.1f}, {:4.1f}, {:4.1f}, {:4.1f}, " \
                                        "{:4.1f}e+19".format(nhi19 * 3, nhi19 * 5, nhi19 * 10, nhi19 * 20,
                                                             nhi19 * 40, nhi19 * 80)
                            # Overlay HI contours on optical image
                            fig = plt.figure(figsize=(8, 8))
                            ax1 = fig.add_subplot(111, projection=WCS(hdulist_opt[0].header))
                            ax1.imshow(d2, cmap='viridis', vmin=np.percentile(d2, 10), vmax=np.percentile(d2, 99.8))
                            ax1.contour(hi_reprojected, cmap='Oranges', levels=[rms * 3, rms * 5, rms * 10, rms * 20,
                                                                                rms * 40, rms * 80])
                            ax1.set_title(name, fontsize=20)
                            ax1.tick_params(axis='both', which='major', labelsize=18)
                            ax1.coords['ra'].set_axislabel('RA (J2000)', fontsize=20)
                            ax1.coords['dec'].set_axislabel('Dec (J2000)', fontsize=20)
                            ax1.text(0.5, 0.05, nhi_label, ha='center', va='center', transform=ax1.transAxes, color='white',
                                     fontsize=18)
                            ax1.add_patch(Ellipse((0.92, 0.9), height=(bmaj/opt_view).decompose(),
                                                  width=(bmin/opt_view).decompose(), angle=bpa, transform=ax1.transAxes,
                                                  edgecolor='white', linewidth=1))

                            fig.savefig(loc + outname + '_{}_overlay.png'.format(int(obj[0])), bbox_inches='tight')
                            hdulist_hi.close()

                        # Make velocity map
                        if not os.path.isfile(loc + outname + '_{}_mom1.png'.format(int(obj[0]))):
                            mom1 = fits.open(loc + outname + '_{}_mom1.fits'.format(int(obj[0])))
                            for i in range(mom1[0].data.shape[0]):
                                for j in range(mom1[0].data.shape[1]):
                                    mom1[0].data[i][j] = (mom1[0].data[i][j] * u.Hz).to(u.km/u.s, equivalencies=optical_HI).value
                                    # Set crazy mom1 values to nan:
                                    if (mom1[0].data[i][j] > velmax) | (mom1[0].data[i][j] < velmin):
                                        mom1[0].data[i][j] = np.nan
                            mom1_reprojected, footprint = reproject_interp(mom1, h2)
                            v_sys_label = "v_sys = {}, W_50 = {}".format(int(v_sys[-1]), int(obj[cathead == "W50"]))
                            fig = plt.figure(figsize=(8, 8))
                            ax1 = fig.add_subplot(111, projection=WCS(hdulist_opt[0].header))
                            im = ax1.imshow(mom1_reprojected, cmap='RdBu_r', vmin=velmin, vmax=velmax)
                            ax1.contour(mom1_reprojected, colors=['white', 'gray', 'black', 'gray', 'white'],
                                        levels=[v_sys[-1]-100, v_sys[-1]-50, v_sys[-1], v_sys[-1]+50, v_sys[-1]+100],
                                        linewidths=0.5)
                            # Plot HI center of galaxy
                            ax1.scatter(c.ra.deg, c.dec.deg, marker='x', c='black', linewidth=0.75, transform=ax.get_transform('icrs'))
                            ax1.set_title(name, fontsize=20)
                            ax1.tick_params(axis='both', which='major', labelsize=18)
                            ax1.coords['ra'].set_axislabel('RA (J2000)', fontsize=20)
                            ax1.coords['dec'].set_axislabel('Dec (J2000)', fontsize=20)
                            ax1.text(0.7, 0.05, v_sys_label, ha='center', va='center', transform=ax1.transAxes,
                                     color='black', fontsize=18)
                            ax1.add_patch(Ellipse((0.92, 0.9), height=(bmaj / opt_view).decompose(), facecolor='gray',
                                                  width=(bmin / opt_view).decompose(), angle=bpa, transform=ax1.transAxes,
                                                  edgecolor='steelblue', linewidth=1))
                            cb_ax = fig.add_axes([0.91, 0.11, 0.02, 0.76])
                            cbar = fig.colorbar(im, cax=cb_ax)
                            cbar.set_label("Velocity [km/s]", fontsize=18)


                            fig.savefig(loc + outname + '_{}_mom1.png'.format(int(obj[0])), bbox_inches='tight')
                            mom1.close()

                # Add derived parameters to objects to then be written to catalog:
                cat['SJyHz'] = SJyHz
                cat['logMhi'] = logMhi
                cat['redshift'] = redshift
                cat['v_sys'] = v_sys
                cat['D_Lum'] = D_Lum
                cat['rms_spec'] = rms_spec
                cat['SNR'] = SNR

                objects = []
                for source in cat:
                    obj = []
                    for s in source:
                        obj.append(s)
                    objects.append(obj)

                # Write out new catalog on a per cube basis:
                write_catalog(objects, catParNames, catParUnits, catParFormt, header, outName=loc + 'final_cat.txt')

                hdu_clean.close()
                hdu_mask3d.close()

            else:
                print("No CLEAN cube for Beam {:02}, Cube {}".format(b, c))
    else:
        print("No clean_cat.txt for Beam {:02}".format(b))

print("Beam info in *specfull.txt can be read from the text files like: a.meta['comments'][0].replace('= ','').split()")
print("[FINALSOURCES] Done.")
