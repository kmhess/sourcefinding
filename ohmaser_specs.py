import os
# from glob import glob

# importing here prevents error messages from apercal
from modules.functions import *

from argparse import ArgumentParser, RawTextHelpFormatter
from astropy.io import ascii, fits
from astropy.convolution import convolve, Box1DKernel
from astropy import units as u
from cosmocalc import cosmocalc
import matplotlib.pyplot as plt
import numpy as np


###################################################################

parser = ArgumentParser(description="Do source finding in the HI spectral line cubes for a given taskid, beam, cubes",
                        formatter_class=RawTextHelpFormatter)

parser.add_argument('-s', '--sourcename', default='AHCJ110246.9+591036_191003042_36_2_1',
                    help='Specify full AHC name of source to plot.  (default: %(default)s).')

###################################################################

# Parse the arguments above
args = parser.parse_args()
name, taskid, b, c, s = args.sourcename.split("_")
b, c, s = int(b), int(c), int(s)
cube_name = 'HI_image_cube'
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
print("\tz = {};  DL_Mpc = {}".format(z,DL))

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

print("[PLOT_OHMASER_SPEC] Grab the 1612 MHz data.")

for c2 in [3, 2, 1, 0]:
    if not os.path.isfile(loc + cube_name + '{}.fits'.format(c2)):
        print("[PLOT_OHMASER_SPEC] Retrieving image cube{} from ALTA.".format(c2))
        os.system('iget {}{}_AP_B0{:02}/HI_image_cube{}.fits {}'.format(alta_dir, taskid, b, c2, loc))
    header = fits.getheader(loc + cube_name + '{}.fits'.format(c2))
    f1 = header['CRVAL3'] * u.Hz
    f2 = (header['CRVAL3'] + header['CDELT3'] * header['NAXIS3']) * u.Hz
    if (f3_extrap > f1) & (f3_extrap < f2):
        print("\tF(1612)obs line in Cube {}".format(c2))
        break

cathead = np.array(cat.colnames)[1:]    # This is to avoid issues with the name column in writeSubcube.
objects = []
for source in oh_candidate:
    obj = []
    for s in source:
        obj.append(s)
    objects.append(obj[1:])
objects = np.array(objects)

# Read in the dirty data covering 1612 MHz line and SoFiA mask from the 1667 MHz cube:
hdu_dirty = fits.open(loc + cube_name + '{}.fits'.format(c2))

pbcor(taskid, loc + cube_name + '{}.fits'.format(c2), hdu_dirty, b, c2)
hdu_pb = pyfits.open(loc + cube_name + '{}_cbcor.fits'.format(c2))

hi_cellsize = hdu_dirty[0].header['CDELT2'] * 3600. * u.arcsec
# pix_per_beam = bmaj/hi_cellsize * bmin/hi_cellsize * np.pi / (4 * np.log(2))
chan_width = hdu_dirty[0].header['CDELT3'] * u.Hz

# Make HI profiles with noise over whole cube by squashing 3D mask:
cube_frequencies = chan2freq(np.array(range(hdu_dirty[0].data.shape[0])), hdu=hdu_dirty)

for obj in objects:
    # Some lines stolen from cubelets in  SoFiA:
    cubeDim = hdu_dirty[0].data.shape
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

    # Array math a lot faster on (spatially) tiny subcubes from cubelets.writeSubcubes:
    subcube = hdu_pb[0].data[:, int(YminNew):int(YmaxNew) + 1, int(XminNew):int(XmaxNew) + 1]
    submask = fits.getdata(loc + args.sourcename + '_mask.fits'.format(int(obj[0])))
    # Can potentially save mask2d as a better nchan if need be because mask values are 0 or 1:
    mask2d = np.sum(submask, axis=0)

    # Calculate spectrum and some fundamental galaxy parameters
    spectrum = np.nansum(subcube[:, mask2d != 0], axis=1)
    specmask = np.zeros(len(spectrum))

    new_outname = loc + args.sourcename + '_1612'

    if not os.path.isfile(new_outname + '_specfull.txt'):
        print("[FINALSOURCES] Making HI spectrum text file for source {}".format(new_outname.split("/")[-1]))
        ascii.write([cube_frequencies, spectrum], new_outname + '_specfull.txt',
                    names=['Frequency [Hz]', 'Flux [Jy/beam*pixel]'])

hdu_dirty.close()
hdu_pb.close()

if not os.path.isfile(new_outname + '_ohmaser_spec.png'):
    print("[PLOT_OHMASER_SPEC] Read in the spectra.")
    spec = ascii.read(loc + args.sourcename + '_specfull.txt')
    spec1612 = ascii.read(new_outname + '_specfull.txt')
    beam_info = spec.meta['comments']
    print("\t", beam_info)

    print("[PLOT_OHMASER_SPEC] Make some plots.")
    smochan = [3, 5]
    box_kernel = Box1DKernel(smochan[0])
    smoothed_data_box = convolve(np.asfarray(spec['Flux [Jy/beam*pixel]']), box_kernel)
    fig, ax = plt.subplots(2, 1, figsize=(15, 8))
    ax[0].plot([spec[0][0], spec[-1][0]], [0, 0], c='gray', linestyle='--')
    ax[0].plot(spec['Frequency [Hz]'], smoothed_data_box, label='')
    ax[0].set_xlim(spec[0][0], spec[-1][0])
    ax[0].plot([f2_obs.to(u.Hz).value, f2_obs.to(u.Hz).value], [-0.1, 0.26], linestyle='--', label='1.667 GHz')
    ax[0].plot([f1_extrap.to(u.Hz).value, f1_extrap.to(u.Hz).value], [-0.1, 0.26], linestyle='--', label='1.665 GHz')
    ax[0].legend()
    ax[0].text(0.02, 0.8, 'Boxcar smoothed by {} channels'.format(smochan[0]), transform=ax[0].transAxes)
    ax[0].text(0.05, 0.9, 'z={:9.6f}'.format(z[0]), transform=ax[0].transAxes)
    box_kernel = Box1DKernel(smochan[1])
    smoothed_data_box = convolve(np.asfarray(spec1612['Flux [Jy/beam*pixel]']), box_kernel)
    ax[1].plot([spec1612[0][0], spec1612[-1][0]], [0, 0], c='gray', linestyle='--')
    ax[1].plot(spec1612['Frequency [Hz]'], smoothed_data_box, label='')
    ax[1].set_xlim(spec1612[0][0], spec1612[-1][0])
    ax[1].plot([f3_extrap.to(u.Hz).value, f3_extrap.to(u.Hz).value], [-0.1, 0.1], linestyle='--', label='1.612 GHz', c='red')
    ax[1].legend()
    ax[1].text(0.02, 0.9, 'Boxcar smoothed by {} channels'.format(smochan[1]), transform=ax[1].transAxes)
    plt.savefig(loc + args.sourcename + '_ohmaser_spec.png', bbox_inches='tight')
