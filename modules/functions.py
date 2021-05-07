import os

from astropy.time import Time
import astropy.units as u
import numpy as np

from .PB_correction_happili2 import *


# ----------------------------------------------
def chan2freq(channels=None, hdu=None):
    frequencies = (channels * hdu[0].header['CDELT3'] + hdu[0].header['CRVAL3']) * u.Hz
    return frequencies


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


# ----------------------------------------------
def plot_flags(flag, ax):
    if flag % 2 != 0:
        ax.text(0.09, 0.90, "!", ha='center', va='center', transform=ax.transAxes,
                 color='orange', fontsize=24, fontweight='bold')
    if flag >= 2:
        ax.text(0.07, 0.90, "!", ha='center', va='center', transform=ax.transAxes,
                color='red', fontsize=24, fontweight='bold')
    return


# ----------------------------------------------
def write_catalog(objects, catHeader, catUnits, catFormat, parList, outName):
    # Determine header sizes based on variable-length formatting
    lenCathead = []
    for j in catFormat: lenCathead.append(
        int(j.split("%")[1].split("e")[0].split("f")[0].split("i")[0].split("d")[0].split(".")[0].split("s")[0]) + 1)

    # Create header
    headerName = ""
    headerUnit = ""
    headerCol = ""
    outFormat = ""
    colCount = 0
    header = "Apertif catalogue written {} UTC.\n".format(Time.now())

    for par in parList:
        index = list(catHeader).index(par)
        headerName += catHeader[index].rjust(lenCathead[index])
        headerUnit += catUnits[index].rjust(lenCathead[index])
        headerCol += ("(%i)" % (colCount + 1)).rjust(lenCathead[index])
        outFormat += catFormat[index] + " "
        colCount += 1
    header += headerName[3:] + '\n' + headerUnit[3:] + '\n' + headerCol[3:]

    # Create catalogue
    outObjects = []
    for obj in objects:
        outObjects.append([])
        for par in parList: outObjects[-1].append(obj[list(catHeader).index(par)])

    # Write ASCII catalogue
    if os.path.isfile(outName):
        header = ''
    with open(outName, 'a') as f:
        np.savetxt(f, np.array(outObjects, dtype=object), fmt=outFormat, header=header)

    return
