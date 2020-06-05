import os
# from glob import glob

# importing here prevents error messages from apercal
from modules.functions import *

from argparse import ArgumentParser, RawTextHelpFormatter
from astropy.io import ascii
from astropy.io import fits as pyfits
from astropy.convolution import convolve, Box1DKernel
from astropy import units as u
from astropy.wcs import WCS
from cosmocalc import cosmocalc
import matplotlib.pyplot as plt
import numpy as np

from apercal.libs import lib
from apercal.subs import managefiles
import apercal


###################################################################

parser = ArgumentParser(description="Do source finding in the HI spectral line cubes for a given taskid, beam, cubes",
                        formatter_class=RawTextHelpFormatter)

parser.add_argument('-s', '--sourcename', default='AHCJ110246.9+591036_191003042_36_2_1',
                    help='Specify full AHC name of source to plot.  (default: %(default)s).')

parser.add_argument('-m', '--move_mask',
                    help='If included, move mask for 1667 to clean 1665 & recreate ALL cubes with *_oh appended.',
                    action='store_true')

parser.add_argument('-o', "--overwrite",
                    help="If option is included, overwrite old clean, model, and residual FITS files.",
                    action='store_true')

###################################################################

# Parse the arguments above
args = parser.parse_args()
name, taskid, b, c, s = args.sourcename.split("_")
b, c, s = int(b), int(c), int(s)
cube_name = 'HI_image_cube'
beam_name = 'HI_beam_cube'
alta_dir = '/altaZone/archive/apertif_main/visibilities_default/'

'''The ground state A-doublet (there are a total of four substates due to hyperfine splitting) 
contains four 18cm transitions at 1612, 1665, 1667, and 1720MHz with relative LTE intensities of 
1:5:9:1 in the optically thin limit.'''
# https://www.narrabri.atnf.csiro.au/observing/spectral.html
f1_rest = 1.6654018 * u.GHz
f2_rest = 1.6673590 * u.GHz
f3_rest = 1.6122310 * u.GHz
f4_rest = 1.7205300 * u.GHz
fhi_rest = 1420405751.77 * u.Hz
fco_rest = 115.271203 * u.GHz
fhcn1_rest = 90.663574 * u.GHz
fhcn2_rest = 88.631847 * u.GHz
# https://ui.adsabs.harvard.edu/abs/2014AAS...22345414H/abstract
f5_rest = 54 * u.MHz
H0 = 70

loc = '/tank/hess/apertif/' + taskid + '/B0' + str(b).zfill(2) + '/'

if os.path.isfile(loc[:-5] + 'final_cat.txt'):
    print("[PLOT_OHMASER_SPEC] Reading source parameters from source finding table, extracting data.")
    cat = ascii.read(loc[:-5] + 'final_cat.txt', header_start=1)
    oh_candidate = cat[(cat['beam'] == b) & (cat['cube'] == c) & (cat['id'] == s)]
    if not oh_candidate:
        print("\tNo source matching those specifications in the final catalog from HI source finding. Try again?")
        exit()
else:
    print("\tFinal catalog from HI source finding doesn't exist for this source. Exiting.")
    exit()

# Calculate central frequency from HI calculated "redshift" & recalculate redshift for OH:
f2_obs = fhi_rest / (oh_candidate['redshift'] + 1)
z = f2_rest/f2_obs - 1
cosmo = cosmocalc(z[0], H0)
DL = cosmo['DL_Mpc'] * u.Mpc
print("\tz = {};  DL_Mpc = {}".format(z, DL))

# Calculate the new redshifed frequencies for interesting lines.
f1_extrap = f1_rest / (z + 1)
f3_extrap = f3_rest / (z + 1)
f4_extrap = f4_rest / (z + 1)
fhi_extrap = fhi_rest / (z + 1)
fco_extrap = fco_rest / (z + 1)
fhcn1_extrap = fhcn1_rest / (z + 1)
fhcn2_extrap = fhcn2_rest / (z + 1)

print("\tIf line is 1667 MHz OH line, then redshifted lines are: ")
print("\t\tF(1665)obs = {}\tF(1612)obs = {}\tF(1720)obs = {}".format(f1_extrap, f3_extrap, f4_extrap))
print("\t\tF(HI)obs = {}\tF(CO)obs = {}\tF(HCN)obs = {}, {}".format(fhi_extrap, fco_extrap, fhcn1_extrap, fhcn2_extrap))

# If SoFiA didn't catch the 1665 line, shift the mask for 1667 to the right velocity, add to old mask and re-clean the dirty cube
if args.move_mask:
    if (not os.path.isfile('{}{}{}_4sig_mask_oh.fits'.format(loc, cube_name, c))) | \
            (not os.path.isfile('{}{}{}_clean_oh_cbcor.fits'.format(loc, cube_name, c))):
        print("[PLOT_OHMASER_SPEC] Copying old mask to new file, and shifting mask to 1665 MHz emission")
        os.system('cp {}{}{}_4sig_mask.fits {}{}{}_4sig_mask_oh.fits'.format(loc, cube_name, c, loc, cube_name, c))
        mask_cube = '{}{}{}_4sig_mask_oh.fits'.format(loc, cube_name, c)
        # Edit mask cube to include 1665 MHz emission
        m = pyfits.open(mask_cube, mode='update')
        mdata = m[0].data
        mshape = m[0].data.shape
        new_src_num = np.max(mdata) + 1
        diff1665 = np.int(np.round((f2_obs - f1_extrap).to(u.Hz) / (m[0].header['CDELT3'] * u.Hz)))
        indices = np.where(mdata != 0)
        for i in range(len(indices[0])):
            if (indices[0][i] >= diff1665) & (mdata[indices[0][i], indices[1][i], indices[2][i]] == s) & \
                    (mdata[indices[0][i]-diff1665, indices[1][i], indices[2][i]] < 1):
                mdata[indices[0][i] - diff1665, indices[1][i], indices[2][i]] = new_src_num
        m[0].data = mdata
        m[0].scale('int16')
        m.flush()
        print("[PLOT_OHMASER_SPEC] Cleaning 1665 MHz emission")

        print("[PLOT_OHMASER_SPEC] Determining the statistics of Beam {:02}, Cube {}.".format(b, c))
        filter_cube = loc + cube_name + '{0}_filtered.fits'.format(c)
        f = pyfits.open(filter_cube)
        mask = np.ones(f[0].data.shape[0], dtype=bool)
        if c == 3: mask[376:662] = False
        lineimagestats = [np.nanmin(f[0].data[mask]), np.nanmax(f[0].data[mask]), np.nanstd(f[0].data[mask])]
        f.close()
        print("\tImage min, max, std: {}".format(lineimagestats[:]))

        # Change to the taskid/beam directory because of the way the clean script works.
        # (Rest of script should be okay b/c paths are explicit.)
        prepare = apercal.prepare()
        managefiles.director(prepare, 'ch', loc)
        # Delete any pre-existing Miriad files.
        os.system('rm -rf model_* beam_* map_* image_* mask_* residual_*')

        print("[PLOT_OHMASER_SPEC] Reading in FITS files, making Miriad mask.")

        line_cube = cube_name + '{0}.fits'.format(c)
        beam_cube = beam_name + '{0}.fits'.format(c)
        mask_cube = cube_name + '{0}_4sig_mask_oh.fits'.format(c)
        mask_expr = '"(<mask_sofia>.eq.-1).or.(<mask_sofia>.eq.' + \
                    ').or.(<mask_sofia>.eq.'.join([str(s), str(new_src_num)]) + ')"'

        fits = lib.miriad('fits')
        fits.op = 'xyin'
        fits.in_ = line_cube
        fits.out = 'map_00'
        fits.go()

        if not os.path.isfile(beam_cube):
            print("[PLOT_OHMASER_SPEC] Retrieving synthesized beam cube from ALTA.")
            os.system('iget {}{}_AP_B0{:02}/HI_beam_cube{}.fits {}'.format(alta_dir, taskid, b, c, loc))
        fits.in_ = beam_cube
        fits.out = 'beam_00'
        fits.go()

        # Work with mask_sofia in current directory...otherwise worried about miriad character length for mask_expr
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
            print("Continuing where I left off")
            print("[PLOT_OHMASER_SPEC] Cleaning OH emission using SoFiA mask for Sources {}.".format(s))
            clean = lib.miriad('clean')
            clean.map = 'map_' + str(minc).zfill(2)
            clean.beam = 'beam_' + str(minc).zfill(2)
            clean.out = 'model_' + str(minc + 1).zfill(2)
            clean.cutoff = lineimagestats[2] * 0.5
            clean.region = '"' + 'mask(mask_' + str(minc).zfill(2) + '/)"'
            clean.go()

            print("[PLOT_OHMASER_SPEC] Restoring line cube.")
            restor = lib.miriad('restor')  # Create the restored image
            restor.model = 'model_' + str(minc + 1).zfill(2)
            restor.beam = 'beam_' + str(minc).zfill(2)
            restor.map = 'map_' + str(minc).zfill(2)
            restor.out = 'image_' + str(minc + 1).zfill(2)
            restor.mode = 'clean'
            restor.go()

            # print("[PLOT_OHMASER_SPEC] Making residual cube.")
            # restor.mode = 'residual'  # Create the residual image
            # restor.out = loc + 'residual_' + str(minc + 1).zfill(2)
            # restor.go()

        if args.overwrite:
            os.system('rm {}_clean_oh.fits {}_residual_oh.fits {}_model_oh.fits'.format(line_cube[:-5], line_cube[:-5],
                                                                                        line_cube[:-5]))

        print("[PLOT_OHMASER_SPEC] Writing out cleaned image, residual, and model to FITS.")
        fits.op = 'xyout'
        fits.in_ = 'image_' + str(minc + 1).zfill(2)
        fits.out = line_cube[:-5] + '_clean_oh.fits'
        fits.go()

        # fits.in_ = loc + 'residual_' + str(minc + 1).zfill(2)
        # fits.out = line_cube[:-5] + '_residual_oh.fits'
        # fits.go()

        fits.in_ = loc + 'model_' + str(minc + 1).zfill(2)
        fits.out = line_cube[:-5] + '_model_oh.fits'
        fits.go()

        # Clean up extra Miriad files
        os.system('rm -rf model_* beam_* map_* image_* mask_* residual_*')

        print("[PLOT_OHMASER_SPEC] Creating CB corrected cube for cleaned 1665 MHz emission.")
        hdu_clean = pyfits.open(loc + cube_name + '{}_clean_oh.fits'.format(c))
        pbcor(taskid, loc + cube_name + '{}_clean_oh.fits'.format(c), hdu_clean, b, c)
        hdu_clean.close()

    else:
        print("\tMask including 1665 MHz emission exists: going to assume cleaning already done!!")

print("[PLOT_OHMASER_SPEC] Grab the 1612 MHz data.")

# Given the observed frequency of 1667 MHz line, find cube where 1612 line should be
for c2 in [3, 2, 1, 0]:
    if not os.path.isfile(loc + cube_name + '{}.fits'.format(c2)):
        print("[PLOT_OHMASER_SPEC] Retrieving image cube{} from ALTA.".format(c2))
        os.system('iget {}{}_AP_B0{:02}/HI_image_cube{}.fits {}'.format(alta_dir, taskid, b, c2, loc))
    header = pyfits.getheader(loc + cube_name + '{}.fits'.format(c2))
    f1 = header['CRVAL3'] * u.Hz
    f2 = (header['CRVAL3'] + header['CDELT3'] * header['NAXIS3']) * u.Hz
    if (f3_extrap > f1) & (f3_extrap < f2):
        print("\tF(1612)obs line in Cube {}".format(c2))
        break

# Read in the dirty data covering 1612 MHz line:
hdu_dirty_1612 = pyfits.open(loc + cube_name + '{}.fits'.format(c2))

# Lines stolen from SoFiA and kept in same format so I don't have to troubleshoot (it's ugly tho)
cathead = np.array(cat.colnames)[1:]    # This is to avoid issues with the name column in writeSubcube.
objects = []
for source in oh_candidate:
    obj = []
    for s in source:
        obj.append(s)
    objects.append(obj[1:])
objects = np.array(objects)

obj = objects[0]
cubeDim = hdu_dirty_1612[0].data.shape
Xc = obj[cathead == "x"][0]
Yc = obj[cathead == "y"][0]
Zc = obj[cathead == "z"][0]
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

submask = pyfits.getdata(loc + args.sourcename + '_mask.fits'.format(int(obj[0])))
mask2d = np.sum(submask, axis=0)

# Create new spectrum files from cleaned data if 1665 MHz line wasn't previously covered
if args.move_mask:
    # Make spectrum file based on sum within the mask
    if not os.path.isfile(loc + args.sourcename + '_oh_specfull.txt'):
        print("[PLOT_OHMASER_SPEC] Creating *_oh_specfull.txt file including cleaned 1665 MHz emission.")
        hdu_pb = pyfits.open(loc + cube_name + '{}_clean_oh_cbcor.fits'.format(c))
        subcube = hdu_pb[0].data[:, int(YminNew):int(YmaxNew) + 1, int(XminNew):int(XmaxNew) + 1]
        spectrum = np.nansum(subcube[:, mask2d != 0], axis=1)
        cube_frequencies = chan2freq(np.array(range(hdu_pb[0].data.shape[0])), hdu=hdu_pb)
        bmaj = hdu_pb[0].header['BMAJ'] * 3600. * u.arcsec
        bmin = hdu_pb[0].header['BMIN'] * 3600. * u.arcsec
        hi_cellsize = hdu_pb[0].header['CDELT2'] * 3600. * u.arcsec
        ascii.write([cube_frequencies, spectrum], loc + args.sourcename + '_oh_specfull.txt',
                    names=['Frequency [Hz]', 'Flux [Jy/beam*pixel]'])
        os.system('echo "# BMAJ = {}\n# BMIN = {}\n# CELLSIZE = {:.2f}" > temp'.format(bmaj, bmin,
                                                                                       hi_cellsize))
        os.system('cat temp ' + loc + args.sourcename + '_oh_specfull.txt' +
                  ' > temp2 && mv temp2 ' + loc + args.sourcename + '_oh_specfull.txt')
        os.system('rm temp')
        hdu_pb.close()
    # Make spectrum file based on central pixel of source
    if not os.path.isfile(loc + args.sourcename + '_oh_pix_specfull.txt'):
        print("[PLOT_OHMASER_SPEC] Creating *_oh_pix_specfull.txt file including cleaned 1665 MHz emission.")
        hdu_pb = pyfits.open(loc + cube_name + '{}_clean_oh_cbcor.fits'.format(c))
        pix_spec = hdu_pb[0].data[:, int(Yc), int(Xc)]
        cube_frequencies = chan2freq(np.array(range(hdu_pb[0].data.shape[0])), hdu=hdu_pb)
        bmaj = hdu_pb[0].header['BMAJ'] * 3600. * u.arcsec
        bmin = hdu_pb[0].header['BMIN'] * 3600. * u.arcsec
        hi_cellsize = hdu_pb[0].header['CDELT2'] * 3600. * u.arcsec
        ascii.write([cube_frequencies, pix_spec], loc + args.sourcename + '_oh_pix_specfull.txt',
                    names=['Frequency [Hz]', 'Flux [Jy/beam*pixel]'])
        os.system('echo "# BMAJ = {}\n# BMIN = {}\n# CELLSIZE = {:.2f}" > temp'.format(bmaj, bmin,
                                                                                       hi_cellsize))
        os.system('cat temp ' + loc + args.sourcename + '_oh_pix_specfull.txt' +
                  ' > temp2 && mv temp2 ' + loc + args.sourcename + '_oh_pix_specfull.txt')
        os.system('rm temp')
        hdu_pb.close()
else:
    print("\tNo extra cleaning necessary, so getting pixel spectrum from original CB corrected file")
    if not os.path.isfile(loc + args.sourcename + '_pix_specfull.txt'):
        print("[PLOT_OHMASER_SPEC] Creating *_pix_specfull.txt file from original CB corrected file.")
        hdu_pb = pyfits.open(loc + cube_name + '{}_clean_cbcor.fits'.format(c))
        pix_spec = hdu_pb[0].data[:, int(Yc), int(Xc)]
        cube_frequencies = chan2freq(np.array(range(hdu_pb[0].data.shape[0])), hdu=hdu_pb)
        bmaj = hdu_pb[0].header['BMAJ'] * 3600. * u.arcsec
        bmin = hdu_pb[0].header['BMIN'] * 3600. * u.arcsec
        hi_cellsize = hdu_pb[0].header['CDELT2'] * 3600. * u.arcsec
        ascii.write([cube_frequencies, pix_spec], loc + args.sourcename + '_pix_specfull.txt',
                    names=['Frequency [Hz]', 'Flux [Jy/beam*pixel]'])
        os.system('echo "# BMAJ = {}\n# BMIN = {}\n# CELLSIZE = {:.2f}" > temp'.format(bmaj, bmin,
                                                                                       hi_cellsize))
        os.system('cat temp ' + loc + args.sourcename + '_oh_specfull.txt' +
                  ' > temp2 && mv temp2 ' + loc + args.sourcename + '_oh_pix_specfull.txt')
        os.system('rm temp')
        hdu_pb.close()

# If dirty cb corrected 1612 MHz doesn't exist, make it.
if not os.path.isfile(loc + cube_name + '{}_cbcor.fits'.format(c2)):
    pbcor(taskid, loc + cube_name + '{}.fits'.format(c2), hdu_dirty_1612, b, c2)
hdu_dirty_1612.close()

# Same for both 1612 and 1667 MHz cubes (unless detected in cube 3...let's not deal with that now.)
hdu_pb = pyfits.open(loc + cube_name + '{}_clean_cbcor.fits'.format(c))
hi_cellsize = hdu_pb[0].header['CDELT2'] * 3600. * u.arcsec
bmaj = hdu_pb[0].header['BMAJ'] * 3600. * u.arcsec
bmin = hdu_pb[0].header['BMIN'] * 3600. * u.arcsec
pix_per_beam = bmaj / hi_cellsize * bmin / hi_cellsize * np.pi / (4 * np.log(2))
chan_width = hdu_pb[0].header['CDELT3'] * u.Hz
hdu_pb.close()
new_outname = loc + args.sourcename + '_1612'

# If spectrum files fo 1612 MHz line don't exist yet, make them
if not os.path.isfile(new_outname + '_specfull.txt'):
    print("[PLOT_OHMASER_SPEC] Creating *_1612_specfull.txt file integrated over 2d mask")
    hdu_pb_1612 = pyfits.open(loc + cube_name + '{}_cbcor.fits'.format(c2))
    # Array math a lot faster on (spatially) tiny subcubes from cubelets.writeSubcubes:
    subcube_1612 = hdu_pb_1612[0].data[:, int(YminNew):int(YmaxNew) + 1, int(XminNew):int(XmaxNew) + 1]
    spectrum_1612 = np.nansum(subcube_1612[:, mask2d != 0], axis=1)
    cube_frequencies = chan2freq(np.array(range(hdu_pb_1612[0].data.shape[0])), hdu=hdu_pb_1612)
    ascii.write([cube_frequencies, spectrum_1612], new_outname + '_specfull.txt',
                names=['Frequency [Hz]', 'Flux [Jy/beam*pixel]'])
    hdu_pb_1612.close()
if not os.path.isfile(new_outname + '_pix_specfull.txt'):
    print("[PLOT_OHMASER_SPEC] Creating *_1612_pix_specfull.txt file through single pixel")
    hdu_pb_1612 = pyfits.open(loc + cube_name + '{}_cbcor.fits'.format(c2))
    pix_spec_1612 = hdu_pb_1612[0].data[:, int(Yc), int(Xc)]
    cube_frequencies = chan2freq(np.array(range(hdu_pb_1612[0].data.shape[0])), hdu=hdu_pb_1612)
    ascii.write([cube_frequencies, pix_spec_1612], new_outname + '_pix_specfull.txt',
                names=['Frequency [Hz]', 'Flux [Jy/beam*pixel]'])
    hdu_pb_1612.close()

# MAKE SOME PLOTS! (Always remake them--it's fast)  CHANGE THIS LATER?
# if not os.path.isfile(new_outname + '_ohmaser_spec.png'):
print("[PLOT_OHMASER_SPEC] Read in the spectra.")
if os.path.isfile(loc + args.sourcename + '_oh_specfull.txt'):
    spec = ascii.read(loc + args.sourcename + '_oh_specfull.txt')
    pix_spec = ascii.read(loc + args.sourcename + '_oh_pix_specfull.txt')
else:
    print("\tWARNING: 1665 MHz may not have been cleaned!!")
    spec = ascii.read(loc + args.sourcename + '_specfull.txt')
    pix_spec = ascii.read(loc + args.sourcename + '_pix_specfull.txt')
# Expect 1612 line to always be weak and not clean-able so file will always be the same.
spec1612 = ascii.read(new_outname + '_specfull.txt')
pix_spec1612 = ascii.read(new_outname + '_pix_specfull.txt')
beam_info = spec.meta['comments']
print("\t", beam_info)

print("[PLOT_OHMASER_SPEC] Make some plots.")
smochan = [3, 5]
box_kernel = Box1DKernel(smochan[0])
smoothed_data_box = convolve(np.asfarray(spec['Flux [Jy/beam*pixel]']), box_kernel)
fig, ax = plt.subplots(2, 1, figsize=(15, 8))
ax[0].plot([spec[0][0], spec[-1][0]], [0, 0], c='gray', linestyle='--')
ax[0].plot(spec['Frequency [Hz]'], smoothed_data_box/pix_per_beam, label='')
ax[0].set_xlim(spec[0][0], spec[-1][0])
ax[0].plot([f2_obs.to(u.Hz).value, f2_obs.to(u.Hz).value], [-0.01, 0.03], linestyle='--', label='1.667 GHz')
ax[0].plot([f1_extrap.to(u.Hz).value, f1_extrap.to(u.Hz).value], [-0.01, 0.03], linestyle='--', label='1.665 GHz')
ax[0].legend()
ax[0].text(0.02, 0.8, 'Boxcar smoothed by {} channels'.format(smochan[0]), transform=ax[0].transAxes)
ax[0].text(0.05, 0.9, 'z={:9.6f}'.format(z[0]), transform=ax[0].transAxes)
ax[0].set_ylabel("Integrated Flux [Jy]")
box_kernel = Box1DKernel(smochan[1])
smoothed_data_box = convolve(np.asfarray(spec1612['Flux [Jy/beam*pixel]']), box_kernel)
ax[1].plot([spec1612[0][0], spec1612[-1][0]], [0, 0], c='gray', linestyle='--')
ax[1].plot(spec1612['Frequency [Hz]'], smoothed_data_box/pix_per_beam, label='')
ax[1].set_xlim(spec1612[0][0], spec1612[-1][0])
ax[1].plot([f3_extrap.to(u.Hz).value, f3_extrap.to(u.Hz).value], [-0.01, 0.01], linestyle='--', label='1.612 GHz', c='red')
ax[1].legend()
ax[1].text(0.02, 0.9, 'Boxcar smoothed by {} channels'.format(smochan[1]), transform=ax[1].transAxes)
ax[1].set_ylabel("Integrated Flux [Jy]")
ax[1].set_xlabel("Observed Frequency [Hz]")
plt.savefig(loc + args.sourcename + '_ohmaser_spec.png', bbox_inches='tight')

box_kernel = Box1DKernel(smochan[0])
smoothed_data_box = convolve(np.asfarray(pix_spec['Flux [Jy/beam*pixel]']), box_kernel)
fig, ax = plt.subplots(2, 1, figsize=(15, 8))
ax[0].plot([pix_spec[0][0], pix_spec[-1][0]], [0, 0], c='gray', linestyle='--')
ax[0].plot(pix_spec['Frequency [Hz]'], smoothed_data_box, label='')
ax[0].set_xlim(pix_spec[0][0], pix_spec[-1][0])
ax[0].plot([f2_obs.to(u.Hz).value, f2_obs.to(u.Hz).value], [-0.005, 0.019], linestyle='--', label='1.667 GHz')
ax[0].plot([f1_extrap.to(u.Hz).value, f1_extrap.to(u.Hz).value], [-0.005, 0.019], linestyle='--', label='1.665 GHz')
ax[0].legend()
ax[0].text(0.02, 0.8, 'Boxcar smoothed by {} channels'.format(smochan[0]), transform=ax[0].transAxes)
ax[0].text(0.05, 0.9, 'z={:9.6f}'.format(z[0]), transform=ax[0].transAxes)
ax[0].set_ylabel("Peak Flux [Jy/beam]")
box_kernel = Box1DKernel(smochan[1])
smoothed_data_box = convolve(np.asfarray(pix_spec1612['Flux [Jy/beam*pixel]']), box_kernel)
ax[1].plot([pix_spec1612[0][0], pix_spec1612[-1][0]], [0, 0], c='gray', linestyle='--')
ax[1].plot(pix_spec1612['Frequency [Hz]'], smoothed_data_box, label='')
ax[1].set_xlim(pix_spec1612[0][0], pix_spec1612[-1][0])
ax[1].plot([f3_extrap.to(u.Hz).value, f3_extrap.to(u.Hz).value], [-0.004, 0.004],
#           [np.nanmax(smoothed_data_box)*-1.05, np.nanmax(smoothed_data_box)*1.05],
           linestyle='--', label='1.612 GHz', c='red')
ax[1].legend()
ax[1].text(0.02, 0.9, 'Boxcar smoothed by {} channels'.format(smochan[1]), transform=ax[1].transAxes)
ax[1].set_ylabel("Peak Flux [Jy/beam]")
ax[0].set_xlabel("Observed Frequency [Hz]")
plt.savefig(loc + args.sourcename + '_ohmaser_pix_spec.png', bbox_inches='tight')
