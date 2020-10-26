import os
from glob import glob
import sys

# importing here prevents error messages from apercal
from modules.functions import *

from argparse import ArgumentParser, RawTextHelpFormatter
from astropy.coordinates import SkyCoord
from astropy import constants as const
from astropy.io import ascii, fits
from astropy import units as u
from astropy.wcs import WCS
from astroquery.skyview import SkyView
from cosmocalc import cosmocalc
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from reproject import reproject_interp
from scipy import ndimage
from scipy.ndimage import generate_binary_structure
# from spectral_cube import SpectralCube

sys.path.insert(0, os.environ['SOFIA_MODULE_PATH'])
from sofia import cubelets


# For testing nearness of spatial issues:
# https://stackoverflow.com/questions/10996769/pixel-neighbors-in-2d-array-image-using-python
def test_mask(value):
    test = np.any(np.isnan(value))
    return test


foot = np.array(generate_binary_structure(2, 1), dtype=int)

###################################################################

parser = ArgumentParser(description="Create new moment maps for (cleaned!) line cubes for a given taskid, beam, cubes",
                        formatter_class=RawTextHelpFormatter)

parser.add_argument('-t', '--taskid', default='190915041',
                    help='Specify the input taskid (default: %(default)s).')

parser.add_argument('-b', '--beams', default='0-39',
                    help='Specify a range (0-39) or list (3,5,7,11) of beams on which to do source finding (default: %(default)s).')

parser.add_argument('-c', '--cubes', default='1,2,3',
                    help='Specify the cubes on which to do source finding (default: %(default)s).')

parser.add_argument('-n', "--nospline",
                    help="No spline fitting has been done; so use clean cube not repaired clean cube.",
                    action='store_true')

# Parse the arguments above
args = parser.parse_args()

# Work on spline fitted (and repaired) data or not:
if args.nospline:
    clean_name = 'clean'
else:
    clean_name = 'rep_clean'

# Range of cubes/beams to work on:
taskid = args.taskid.replace("/", "")
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
catParUnits = ["-", "-", "pix", "pix", "chan", "pix", "pix", "pix", "pix", "chan", "chan", "-",
               "Jy/beam", "Jy/beam", "Jy/beam", "-", "-", "Jy/beam", "km/s", "km/s", "pix", "pix", "pix",
               "pix", "pix", "deg", "deg", "-", "-", "-",
               "Jy*Hz", "log(M_Sun)", "-", "km/s", "Mpc", "Jy/chan", "-"]
catParFormt = ("%18s", "%7i", "%10.3f", "%10.3f", "%10.3f", "%7i", "%7i", "%7i", "%7i", "%7i", "%7i", "%8i",
               "%10.7f", "%10.7f", "%12.6f", "%8.6f", "%7i", "%12.6f", "%10.3f", "%10.3f", "%10.3f", "%10.3f", "%10.3f",
               "%10.3f", "%10.3f", "%10.3f", "%10.3f", "%10i", "%7i", "%7i",
               "%13.6f", "%12.6f", "%11.7f", "%11.3f", "%10.3f", "%11.7f", "%8.3f")

for b in beams:
    loc = '/tank/hess/apertif/' + taskid + '/B0' + str(b).zfill(2) + '/'
    if os.path.isfile(loc + '{}_cat.txt'.format(clean_name)):
        # Read in the master catalog of cleaned sources
        catalog = ascii.read(loc + '{}_cat.txt'.format(clean_name), header_start=1)

        for c in cubes:
            if os.path.isfile(loc + cube_name + '{}_{}.fits'.format(c, clean_name)) | \
                    os.path.isfile(loc + cube_name + '{}_{}_cbcor.fits'.format(c, clean_name)):
                cat = catalog[catalog['cube'] == c]
                cathead = np.array(cat.colnames)[1:]    # This is to avoid issues with the name column in writeSubcube.
                print("\tFound {} sources in Beam {:02} Cube {}".format(len(cat), b, c))

                # Read in the cleaned data and original SoFiA mask:
                if os.path.isfile(loc + cube_name + '{}_{}_cbcor.fits'.format(c, clean_name)):
                    hdu_clean = fits.open(loc + cube_name + '{}_{}_cbcor.fits'.format(c, clean_name))
                else:
                    hdu_clean = fits.open(loc + cube_name + '{}_{}.fits'.format(c, clean_name))
                hdu_mask3d = fits.open(loc + cube_name + '{}_4sig_mask.fits'.format(c))
                hdu_filter = fits.open(loc + cube_name + '{}_filtered.fits'.format(c))

                pbcor(taskid, loc + cube_name + '{}_{}.fits'.format(c, clean_name), hdu_clean, b, c)
                hdu_pb = fits.open(loc + cube_name + '{}_{}_cbcor.fits'.format(c, clean_name))

                outname = 'X_{}_{:02}_{}'.format(taskid, b, c)
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
                print("[FINALSOURCES] Making cubelets for sources in clean_cat.txt Beam {:02} Cube {}".format(b, c))
                cubelets.writeSubcube(hdu_pb[0].data, hdu_clean[0].header, hdu_mask3d[0].data, objects, cathead,
                                      outname, loc, False, False)

                # Get beam size and cell size
                bmaj = hdu_clean[0].header['BMAJ'] * 3600. * u.arcsec
                bmin = hdu_clean[0].header['BMIN'] * 3600. * u.arcsec
                bpa = hdu_clean[0].header['BPA']
                hi_cellsize = hdu_clean[0].header['CDELT2'] * 3600. * u.arcsec
                pix_per_beam = bmaj/hi_cellsize * bmin/hi_cellsize * np.pi / (4 * np.log(2))
                chan_width = hdu_clean[0].header['CDELT3'] * u.Hz
                opt_view = 6. * u.arcmin
                opt_pixels = 900

                # Make HI profiles with noise over whole cube by squashing 3D mask:
                cube_frequencies = chan2freq(np.array(range(hdu_clean[0].data.shape[0])), hdu=hdu_clean)

                SJyHz, logMhi, redshift, v_sys, D_Lum, rms_spec, SNR, w50, w20, src_name = [], [], [], [], [], [], [], [], [], []
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
                    cPixZNew = int(Zc)
                    maxX = 2 * max(abs(cPixXNew - Xmin), abs(cPixXNew - Xmax))
                    maxY = 2 * max(abs(cPixYNew - Ymin), abs(cPixYNew - Ymax))
                    maxZ = 2 * max(abs(cPixZNew - Zmin), abs(cPixZNew - Zmax))
                    XminNew = cPixXNew - maxX
                    if XminNew < 0: XminNew = 0
                    YminNew = cPixYNew - maxY
                    if YminNew < 0: YminNew = 0
                    ZminNew = cPixZNew - maxZ
                    if ZminNew < 0: ZminNew = 0
                    XmaxNew = cPixXNew + maxX
                    if XmaxNew > cubeDim[2] - 1: XmaxNew = cubeDim[2] - 1
                    YmaxNew = cPixYNew + maxY
                    if YmaxNew > cubeDim[1] - 1: YmaxNew = cubeDim[1] - 1
                    ZmaxNew = cPixZNew + maxZ
                    if ZmaxNew > cubeDim[0] - 1: ZmaxNew = cubeDim[0] - 1

                    # Do some prep for mom1 maps:
                    freqmin = chan2freq(Zmin, hdu_pb)
                    freqmax = chan2freq(Zmax, hdu_pb)
                    velmax = freqmin.to(u.km/u.s, equivalencies=optical_HI).value
                    velmin = freqmax.to(u.km/u.s, equivalencies=optical_HI).value
                    kinpa = obj[cathead == "kin_pa"][0] * u.deg
                    rms_sofia = obj[cathead == "rms"][0]

                    # Array math a lot faster on (spatially) tiny subcubes from cubelets.writeSubcubes:
                    subcube = hdu_pb[0].data[:, int(YminNew):int(YmaxNew) + 1, int(XminNew):int(XmaxNew) + 1]
                    submask = fits.getdata(loc + outname + '_{}_mask.fits'.format(int(obj[0])))
                    # Can potentially save mask2d as a better nchan if need be because mask values are 0 or 1:
                    mask2d = np.sum(submask, axis=0)
                    # Create subimage of the continuum filtering to raise flag if it affects source
                    filter2d = hdu_filter[0].data[0, int(YminNew):int(YmaxNew) + 1, int(XminNew):int(XmaxNew) + 1]

                    # Calculate spectrum and some fundamental galaxy parameters
                    spectrum = np.nansum(subcube[:, mask2d != 0], axis=1)
                    signal = np.nansum(spectrum[int(Zmin):int(Zmax)])
                    SJyHz.append(signal * chan_width.value / pix_per_beam)
                    freq_sys = chan2freq(channels=int(obj[cathead == "z"][0]), hdu=hdu_clean)  # Remove int at some point!!!
                    # w50 & w20 in rest frame of the observer
                    w50.append((const.c * obj[cathead == "w50"][0] * chan_width / freq_sys).to(u.km/u.s).value)
                    w20.append((const.c * obj[cathead == "w20"][0] * chan_width / freq_sys).to(u.km/u.s).value)
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

                    # Generate some flags based on AAS filter (1) or continuum filtering (2)
                    flag = 0
                    if np.any(spectrum[specmask == 1] == 0.0):
                        flag += 1
                        print("\tSpectrum Flag")
                    result = ndimage.generic_filter(filter2d, test_mask, footprint=foot)
                    if np.sum(result * mask2d) > 0:
                        flag += 2
                        print("\tSpatial filtering flag")

                    # Determine HI position of galaxy & therefore source name
                    subcoords = wcs.wcs_pix2world(Xc, Yc, 1, 0)
                    hi_pos = SkyCoord(ra=subcoords[0], dec=subcoords[1], unit=u.deg)
                    src_name.append('AHC J{0}{1}'.format(hi_pos.ra.to_string(unit=u.hourangle, sep='', precision=1, pad=True),
                                                    hi_pos.dec.to_string(sep='', precision=0, alwayssign=True, pad=True)))

                    # Having determined source coordinate based name, rename cubelet products:
                    cubelet_products = glob(loc + outname + "_" + str(int(obj[0])) + '_*')
                    cubelet_products += glob(loc + outname + "_" + str(int(obj[0])) + '.fits')
                    mv_to_name = loc + "AHC" + src_name[-1].split(" ")[1]
                    for p in cubelet_products:
                        os.system("mv " + p + " " + mv_to_name + p.split("X")[-1])
                    new_outname = loc + "AHC" + src_name[-1].split(" ")[1] + outname[1:] + "_" + str(int(obj[0]))

                    # Get optical image
                    path = SkyView.get_images(position=hi_pos.to_string('hmsdms'), width=opt_view, height=opt_view,
                                              survey=['DSS2 Blue'], pixels=opt_pixels)

                    if len(path) != 0:
                        print("[FINALSOURCES] Optical image retrieved from DSS2 Blue")
                        # Get optical image and HI subimage for object
                        hdulist_opt = path[0]
                        d2 = hdulist_opt[0].data
                        h2 = hdulist_opt[0].header

                        # Make a total intensity map overlayed on optical, HI grey scale, and HI significance maps
                        if (not os.path.isfile(new_outname + '_mom0.png')) | (
                        not os.path.isfile(new_outname + '_mom0hi.png')) | (
                        not os.path.isfile(new_outname + '_signif.png')):
                            hdulist_hi = fits.open(new_outname + '_mom0.fits')
                            # Reproject HI data & calculate contour properties
                            hi_reprojected, footprint = reproject_interp(hdulist_hi, h2)
                            # Calculate noise over narrower range to avoid bad spws
                            subsubcube = subcube[int(ZminNew):int(ZmaxNew) + 1, :, :]
                            rms = np.nanstd(subsubcube[submask == 0]) * chan_width.value
                            hdulist_mask2d = fits.PrimaryHDU(mask2d, hdulist_hi[0].header)
                            mask2d_reprojected, footprint = reproject_interp(hdulist_mask2d, h2)
                            significance = hi_reprojected/(rms * np.sqrt(mask2d_reprojected))
                            sensitivity = np.median(hi_reprojected[(significance>=2)*(significance<=3)])
                            nhi19 = 2.33e20 * rms / (bmaj.value * bmin.value) / 1e19 # 1 sigma
                            print("\t1sig N_HI is {}e+19".format(nhi19))
                            nhi_label = "N_HI = {:.1f}, {:.1f}, {:.1f}, {:.0f}, " \
                                        "{:.0f}e+19".format(nhi19 * 3, nhi19 * 5, nhi19 * 10, nhi19 * 20, nhi19 * 40) #, nhi19 * 80)

                            # Overlay HI contours on optical image
                            if not os.path.isfile(new_outname + '_mom0.png'):
                                print("[FINALSOURCES] Making optical overlay for source {}".format(new_outname.split("/")[-1]))
                                fig = plt.figure(figsize=(8, 8))
                                ax1 = fig.add_subplot(111, projection=WCS(hdulist_opt[0].header))
                                ax1.imshow(d2, cmap='viridis', vmin=np.percentile(d2, 10), vmax=np.percentile(d2, 99.8), origin='lower')
                                ax1.contour(hi_reprojected, cmap='Oranges', linewidth=0.8,
                                            levels=sensitivity*2**np.range(10))
                                ax1.scatter(hi_pos.ra.deg, hi_pos.dec.deg, marker='x', c='black', linewidth=0.75,
                                            transform=ax1.get_transform('fk5'))
                                ax1.set_title(src_name[-1], fontsize=20)
                                ax1.tick_params(axis='both', which='major', labelsize=18)
                                ax1.coords['ra'].set_axislabel('RA (J2000)', fontsize=20)
                                ax1.coords['dec'].set_axislabel('Dec (J2000)', fontsize=20)
                                ax1.text(0.5, 0.05, nhi_label, ha='center', va='center', transform=ax1.transAxes,
                                         color='white', fontsize=18)
                                ax1.add_patch(Ellipse((0.92, 0.9), height=(bmaj/opt_view).decompose(),
                                                      width=(bmin/opt_view).decompose(), angle=bpa, transform=ax1.transAxes,
                                                      edgecolor='white', linewidth=1))
                                if flag != 0: plot_flags(flag, ax1)
                                fig.savefig(new_outname + '_mom0.png', bbox_inches='tight')

                            # Make HI grey scale image
                            if not os.path.isfile(new_outname + '_mom0hi.png'):
                                print("[FINALSOURCES] Making HI grey scale for source {}".format(
                                    new_outname.split("/")[-1]))
                                fig = plt.figure(figsize=(8, 8))
                                ax1 = fig.add_subplot(111, projection=WCS(hdulist_opt[0].header))
                                im = ax1.imshow(hi_reprojected, cmap='gray_r', origin='lower')
                                ax1.contour(hi_reprojected, cmap='Oranges_r', linewidth=0.8,
                                            levels=[rms * 3, rms * 5, rms * 10, rms * 20, rms * 40])  # , rms * 80]
                                ax1.scatter(hi_pos.ra.deg, hi_pos.dec.deg, marker='x', c='white', linewidth=0.75,
                                            transform=ax1.get_transform('fk5'))
                                ax1.set_title(src_name[-1], fontsize=20)
                                ax1.tick_params(axis='both', which='major', labelsize=18)
                                ax1.coords['ra'].set_axislabel('RA (J2000)', fontsize=20)
                                ax1.coords['dec'].set_axislabel('Dec (J2000)', fontsize=20)
                                ax1.text(0.5, 0.05, nhi_label, ha='center', va='center', transform=ax1.transAxes,
                                         fontsize=18)
                                ax1.add_patch(Ellipse((0.92, 0.9), height=(bmaj / opt_view).decompose(),
                                                      width=(bmin / opt_view).decompose(), angle=bpa,
                                                      transform=ax1.transAxes, facecolor='darkorange',
                                                      edgecolor='black', linewidth=1))
                                if flag != 0: plot_flags(flag, ax1)
                                cb_ax = fig.add_axes([0.91, 0.11, 0.02, 0.76])
                                cbar = fig.colorbar(im, cax=cb_ax)
                                cbar.set_label("HI Intensity [Jy/beam*Hz]", fontsize=18)
                                fig.savefig(new_outname + '_mom0hi.png', bbox_inches='tight')

                            # Make HI significance image
                            if not os.path.isfile(new_outname + '_signif.png'):
                                print("[FINALSOURCES] Making HI significance image for source {}".format(new_outname.split("/")[-1]))
                                fig = plt.figure(figsize=(8, 8))
                                ax1 = fig.add_subplot(111, projection=WCS(hdulist_opt[0].header))
                                im = ax1.imshow(significance, cmap='plasma', origin='lower')
                                ax1.contour(significance, cmap='viridis', linewidth=0.7,
                                            levels=[1, 2, 3, 4, 6, 8, 12])
                                ax1.scatter(hi_pos.ra.deg, hi_pos.dec.deg, marker='x', c='black', linewidth=0.75,
                                            transform=ax1.get_transform('fk5'))
                                ax1.set_title(src_name[-1], fontsize=20)
                                ax1.tick_params(axis='both', which='major', labelsize=18)
                                ax1.coords['ra'].set_axislabel('RA (J2000)', fontsize=20)
                                ax1.coords['dec'].set_axislabel('Dec (J2000)', fontsize=20)
                                ax1.text(0.5, 0.05, '1, 2, 3, 4, 6, 8 * $\Sigma$', ha='center', va='center',
                                         transform=ax1.transAxes, fontsize=18)
                                ax1.add_patch(Ellipse((0.92, 0.9), height=(bmaj / opt_view).decompose(),
                                                      width=(bmin / opt_view).decompose(), angle=bpa,
                                                      transform=ax1.transAxes, facecolor='gold',
                                                      edgecolor='indigo', linewidth=1))
                                if flag != 0: plot_flags(flag, ax1)
                                cb_ax = fig.add_axes([0.91, 0.11, 0.02, 0.76])
                                cbar = fig.colorbar(im, cax=cb_ax)
                                cbar.set_label("Significance", fontsize=18)
                                fig.savefig(new_outname + '_signif.png', bbox_inches='tight')
                            hdulist_hi.close()

                        # Make velocity map for object
                        if not os.path.isfile(new_outname + '_mom1.png'):
                            print("[FINALSOURCES] Making velocity map for source {}".format(new_outname.split("/")[-1]))
                            mom1 = fits.open(new_outname + '_mom1.fits')
                            for i in range(mom1[0].data.shape[0]):
                                for j in range(mom1[0].data.shape[1]):
                                    mom1[0].data[i][j] = (mom1[0].data[i][j] * u.Hz).to(u.km/u.s, equivalencies=optical_HI).value
                                    # Set crazy mom1 values to nan:
                                    if (mom1[0].data[i][j] > velmax) | (mom1[0].data[i][j] < velmin):
                                        mom1[0].data[i][j] = np.nan
                            mom1_reprojected, footprint = reproject_interp(mom1, h2)
                            v_sys_label = "v_sys = {}   W_50 = {}  W_20 = {}".format(int(v_sys[-1]), int(w50[-1]), int(w20[-1]))
                            fig = plt.figure(figsize=(8, 8))
                            ax1 = fig.add_subplot(111, projection=WCS(hdulist_opt[0].header))
                            im = ax1.imshow(mom1_reprojected, cmap='RdBu_r', vmin=velmin, vmax=velmax, origin='lower')
                            if velmax - velmin > 200:
                                levels = [v_sys[-1] - 100, v_sys[-1] - 50, v_sys[-1], v_sys[-1] + 50, v_sys[-1] + 100]
                                clevels = ['white', 'gray', 'black', 'gray', 'white']
                            else:
                                levels = [v_sys[-1] - 50, v_sys[-1], v_sys[-1] + 50]
                                clevels = ['lightgray', 'black', 'lightgray']
                            ax1.contour(mom1_reprojected, colors=clevels, levels=levels, linewidths=0.6)
                            # Plot HI center of galaxy
                            ax1.scatter(hi_pos.ra.deg, hi_pos.dec.deg, marker='x', c='black', linewidth=0.75,
                                        transform=ax1.get_transform('fk5'))
                            ax1.plot([(hi_pos.ra + 0.5*opt_view * np.sin(kinpa)/np.cos(hi_pos.dec)).deg,
                                      (hi_pos.ra - 0.5*opt_view * np.sin(kinpa)/np.cos(hi_pos.dec)).deg],
                                     [(hi_pos.dec + 0.5*opt_view * np.cos(kinpa)).deg, (hi_pos.dec - 0.5*opt_view * np.cos(kinpa)).deg],
                                     c='black', linestyle='--', linewidth=0.75, transform=ax1.get_transform('fk5'))
                            # ax1.scatter([(hi_pos.ra + 0.6 * opt_view * np.sin(kinpa)).deg],[]
                            ax1.set_title(src_name[-1], fontsize=20)
                            ax1.tick_params(axis='both', which='major', labelsize=18)
                            ax1.coords['ra'].set_axislabel('RA (J2000)', fontsize=20)
                            ax1.coords['dec'].set_axislabel('Dec (J2000)', fontsize=20)
                            ax1.text(0.5, 0.05, v_sys_label, ha='center', va='center', transform=ax1.transAxes,
                                     color='black', fontsize=18)
                            ax1.add_patch(Ellipse((0.92, 0.9), height=(bmaj / opt_view).decompose(), facecolor='gray',
                                                  width=(bmin / opt_view).decompose(), angle=bpa,
                                                  transform=ax1.transAxes, edgecolor='steelblue', linewidth=1))
                            if flag != 0: plot_flags(flag, ax1)
                            cb_ax = fig.add_axes([0.91, 0.11, 0.02, 0.76])
                            cbar = fig.colorbar(im, cax=cb_ax)
                            cbar.set_label("Velocity [km/s]", fontsize=18)
                            fig.savefig(new_outname + '_mom1.png', bbox_inches='tight')
                            mom1.close()
                        hdulist_opt.close()
                    else:
                        print("\tWARNING: No optical image found, so no moment-related png's produced")

                    # Make pv plot for object
                    if not os.path.isfile(new_outname + '_pv.png'):
                        print("[FINALSOURCES] Making pv slice for source {}".format(new_outname.split("/")[-1]))
                        pv = fits.open(new_outname + '_pv.fits')
                        wcs_pv = WCS(pv[0].header)
                        ang1, freq1 = wcs_pv.wcs_pix2world(0, 0, 0)
                        ang2, freq2 = wcs_pv.wcs_pix2world(pv[0].header['NAXIS1'], pv[0].header['NAXIS2'], 0)
                        pv_rms = np.nanstd(pv[0].data)
                        fig = plt.figure(figsize=(8, 8))
                        ax1 = fig.add_subplot(111, projection=WCS(pv[0].header))
                        ax1.imshow(pv[0].data, cmap='gray')
                        if np.all(np.isnan(pv[0].data)): continue
                        ax1.contour(pv[0].data, colors='black', levels=[-2*pv_rms, 2*pv_rms, 4*pv_rms])
                        ax1.autoscale(False)
                        ax1.plot([0.0, 0.0], [freq1, freq2], c='orange', linestyle='--', linewidth=0.75,
                                 transform=ax1.get_transform('world'))
                        ax1.plot([ang1, ang2], [freq_sys.value, freq_sys.value], c='orange', linestyle='--',
                                 linewidth=0.75, transform=ax1.get_transform('world'))
                        ax1.set_title(src_name[-1], fontsize=16)
                        ax1.tick_params(axis='both', which='major', labelsize=18)
                        ax1.set_xlabel('Angular Offset', fontsize=16)
                        ax1.set_ylabel('Frequency [Hz]', fontsize=16)
                        ax1.coords[1].set_ticks_position('l')
                        freq_yticks = ax1.get_yticks()  # freq auto yticks from matplotlib
                        ax2 = ax1.twinx()
                        vel1 = const.c.to(u.km/u.s).value * (HI_restfreq.value / freq1 - 1)
                        vel2 = const.c.to(u.km/u.s).value * (HI_restfreq.value / freq2 - 1)
                        ax2.set_ylim(vel1, vel2)
                        ax2.set_ylabel('Velocity [km/s]')
                        ax1.text(0.5, 0.05, 'Kinematic PA = {:5.1f} deg'.format(kinpa.value), ha='center', va='center',
                                 transform=ax1.transAxes, color='white', fontsize=18)
                        fig.savefig(new_outname + '_pv.png', bbox_inches='tight')
                        pv.close()

                    # Save spectrum to a txt file:
                    if not os.path.isfile(new_outname + '_specfull.txt'):
                        print("[FINALSOURCES] Making HI spectrum text file for source {}".format(new_outname.split("/")[-1]))
                        ascii.write([cube_frequencies, spectrum],
                                    new_outname + '_specfull.txt',
                                    names=['Frequency [Hz]', 'Flux [Jy/beam*pixel]'])
                        os.system('echo "# BMAJ = {}\n# BMIN = {}\n# CELLSIZE = {:.2f}" > temp'.format(bmaj, bmin,
                                                                                                       hi_cellsize))
                        os.system('cat temp ' + new_outname + '_specfull.txt' +
                                  ' > temp2 && mv temp2 ' + new_outname + '_specfull.txt')
                        os.system('rm temp')

                    # Make spectrum plot:
                    if not os.path.isfile(new_outname + '_specfull.png'):
                        print("[FINALSOURCES] Making HI spectrum plot for source {}".format(new_outname.split("/")[-1]))
                        spectrumJy = spectrum / pix_per_beam
                        cube_frequencies = chan2freq(np.array(range(hdu_clean[0].data.shape[0])), hdu=hdu_clean)
                        optical_velocity = cube_frequencies.to(u.km / u.s, equivalencies=optical_HI)
                        maskmin = chan2freq(Zmin, hdu=hdu_filter).to(u.km / u.s, equivalencies=optical_HI).value
                        maskmax = chan2freq(Zmax, hdu=hdu_filter).to(u.km / u.s, equivalencies=optical_HI).value
                        fig = plt.figure(figsize=(15, 4))
                        ax_spec = fig.add_subplot(111)
                        ax_spec.plot([optical_velocity[-1].value, optical_velocity[0].value], [0, 0], '--', color='gray')
                        ax_spec.plot(optical_velocity, spectrumJy)
                        ax_spec.plot([maskmin, maskmin], [np.nanmin(spectrumJy), np.nanmax(spectrumJy)], ':', color='gray')
                        ax_spec.plot([maskmax, maskmax], [np.nanmin(spectrumJy), np.nanmax(spectrumJy)], ':', color='gray')
                        ax_spec.set_title(src_name[-1])
                        ax_spec.set_xlim(optical_velocity[-1].value, optical_velocity[0].value)
                        if (np.max(spectrumJy) > 2.) | (np.min(spectrumJy) < -1.):
                            ax_spec.set_ylim(np.max(spectrumJy[int(Zmin):int(Zmax)]) * -2,
                                             np.max(spectrumJy[int(Zmin):int(Zmax)]) * 2)
                        ax_spec.set_ylabel("Integrated Flux [Jy]")
                        ax_spec.set_xlabel("Optical Velocity [km/s]")
                        fig.savefig(new_outname + '_specfull.png', bbox_inches='tight')

                    # Make SoFiA spectrum plot (no noise):
                    if not os.path.isfile(new_outname + '_spec.png'):
                        print("[FINALSOURCES] Making HI SoFiA spectrum plot for source {}".format(new_outname.split("/")[-1]))
                        spec = ascii.read(new_outname + '_spec.txt', names=['Chan', 'Spectral', 'Sum', 'Npix'])
                        optical_velocity = (spec['Spectral'] * u.Hz).to(u.km / u.s, equivalencies=optical_HI)
                        fig = plt.figure(figsize=(8, 4))
                        ax_spec = fig.add_subplot(111)
                        ax_spec.plot([optical_velocity[-1].value, optical_velocity[0].value], [0, 0], '--', color='gray')
                        ax_spec.errorbar(optical_velocity[:].value, spec['Sum']/pix_per_beam, elinewidth=0.75,
                                         yerr=rms_sofia*np.sqrt(spec['Npix']/pix_per_beam), capsize=1)
                        ax_spec.set_title(src_name[-1])
                        ax_spec.set_xlim(optical_velocity[-1].value, optical_velocity[0].value)
                        ax_spec.set_ylabel("Integrated Flux [Jy]")
                        ax_spec.set_xlabel("Optical Velocity [km/s]")
                        fig.savefig(new_outname + '_spec.png', bbox_inches='tight')

                    plt.close('all')

                # Add derived parameters for objects in cube to then be written to catalog:
                cat['SJyHz'] = SJyHz
                cat['logMhi'] = logMhi
                cat['redshift'] = redshift
                cat['v_sys'] = v_sys
                cat['D_Lum'] = D_Lum
                cat['rms_spec'] = rms_spec
                cat['SNR'] = SNR

                # Replace these for their km/s values instead of pixel values (Catalog units hard coded above).
                cat['w50'] = w50
                cat['w20'] = w20
                
                cat['name'] = cat['name'].astype('|S18')
                src_name = [srcname.split(" ")[1] for srcname in src_name]
                cat['name'] = src_name

                objects = []
                for source in cat:
                    obj = []
                    for s in source:
                        obj.append(s)
                    objects.append(obj)

                # Write out new or update catalog on a per cube basis:
                print("[FINALSOURCES] Updating HI final_cat.txt for Beam {:02} Cube {} in taskid directory".format(b, c))
                write_catalog(objects, catParNames, catParUnits, catParFormt, header, outName=loc[:-5] + 'final_cat.txt')
                # write_catalog(objects, catParNames, catParUnits, catParFormt, header, outName='final_cat.txt')

                # Close all related cube files
                hdu_clean.close()
                hdu_mask3d.close()
                hdu_filter.close()
                hdu_pb.close()

            else:
                print("\tNo CLEAN cube for Beam {:02}, Cube {}".format(b, c))
    else:
        print("\tNo {}_cat.txt for Beam {:02}".format(clean_name, b))

print("\tBeam info in *specfull.txt can be read from the text files like: a.meta['comments'][0].replace('= ','').split()")
print("[FINALSOURCES] Done.")
