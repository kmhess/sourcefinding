import os

from astropy.time import Time
import astropy.units as u
import numpy as np

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
    header = "Apertif catalogue written {}.\n" % Time.now()

    for par in parList:
        index = list(catHeader).index(par)
        headerName += catHeader[index].rjust(lenCathead[index])
        headerUnit += catUnits[index].rjust(lenCathead[index])
        headerCol += ("(%i)" % (colCount + 1)).rjust(lenCathead[index])
        outFormat += catFormat[index] + " "
        colCount += 1
    header += headerName[3:] + '\n' + headerUnit[3:] + '\n' + headerCol[3:]
    print(outFormat)

    # Create catalogue
    outObjects = []
    for obj in objects:
        outObjects.append([])
        for par in parList: outObjects[-1].append(obj[list(catHeader).index(par)])
    print(outObjects)

    # Write ASCII catalogue
    if os.path.isfile(outName):
        header = ''
    with open(outName, 'a') as f:
        np.savetxt(f, np.array(outObjects, dtype=object), fmt=outFormat, header=header)

    return