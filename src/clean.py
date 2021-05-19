from __future__ import print_function
import logging
import os

from modules.natural_cubic_spline import fspline

from argparse import ArgumentParser, RawTextHelpFormatter
from astropy.io import ascii
import astropy.io.fits as pyfits
import numpy as np

from multiprocessing import Queue, Process, cpu_count
from tqdm.auto import trange

from apercal.libs import lib
from apercal.subs import managefiles
import apercal

from modules.functions import write_catalog


def worker(inQueue, outQueue):

    """
    Defines the worker process of the parallelisation with multiprocessing.Queue
    and multiprocessing.Process.
    """
    for i in iter(inQueue.get, 'STOP'):

        status = run(i)

        outQueue.put(( status ))


def run(i):
    global new_splinecube_data

    try:
        # Do the spline fitting on the z-axis to masked cube
        # First, replace the previous continuum filtered pixels with spline fitted values, without masking:
        if np.isnan(filtered_pixels[x[i], y[i]]):  # | (str(mask2d[x[i], y[i]]) not in sources):
            # print('continuum', end=" ")
            temp = np.copy(orig_data[:, x[i], y[i]])
            fit = fspline(np.linspace(1, orig_data.shape[0], orig_data.shape[0]), np.nan_to_num(temp), k=5)
            new_splinecube_data[:, x[i], y[i]] = temp - fit
        # Second, use source mask to undo potential over subtraction:
        if str(mask2d[x[i], y[i]]) in sources:
            # print('hi', end=" ")
            s = mask2d[x[i], y[i]]
            zmin, zmax = np.int(catalog[catalog['id'] == s]['z_min']), np.int(catalog[catalog['id'] == s]['z_max'])
            # print(zmin, zmax, end=" ")
            temp = np.copy(orig_data[:, x[i], y[i]])
            # Currently nan --> 0, but could try N (10) nearest neighbors instead...
            # This also doesn't deal with multiple sources along the line of sight...(but then can't do N-nn)
            temp[zmin:zmax] = np.nan
            fit = fspline(np.linspace(1, orig_data.shape[0], orig_data.shape[0]), np.nan_to_num(temp), k=12)
            new_splinecube_data[:, x[i], y[i]] = orig_data[:, x[i], y[i]] - fit
        return 'OK'

    except Exception:
        print("[ERROR] Something went wrong with the Spline fitting [" + str(i) + "]")
        return np.nan
###################################################################

parser = ArgumentParser(description="Do source finding in the HI spectral line cubes for a given taskid, beam, cubes",
                        formatter_class=RawTextHelpFormatter)

parser.add_argument('-t', '--taskid', default='190915041',
                    help='Specify the input taskid (default: %(default)s).')

parser.add_argument('-b', '--beams', default='0-39',
                    help='Specify a range (0-39) or list (3,5,7,11) of beams on which to do source finding (default: %(default)s).')

parser.add_argument('-c', '--cubes', default='1,2,3',
                    help='Specify the cubes on which to do source finding (default: %(default)s).')

parser.add_argument('-s', '--sources', default='all',
                    help='Specify sources to clean.  Can specify range or list. (default: %(default)s).')

parser.add_argument('-n', "--nospline",
                    help="Don't do spline fitting; so source finding on only continuum filtered cube.",
                    action='store_true')

parser.add_argument('-o', "--overwrite",
                    help="If option is included, overwrite old clean, model, and residual FITS files and 'repaired' spline file.",
                    action='store_true')

parser.add_argument('-j', "--njobs",
                    help="Number of jobs to run in parallel (default: %(default)s) tested on happili-05.",
                    default=18)

# Parse the arguments above
args = parser.parse_args()
njobs = int(args.njobs)

###################################################################

# Range of cubes/beams to work on:
taskid = args.taskid
cubes = [int(c) for c in args.cubes.split(',')]

if '-' in args.beams:
    b_range = args.beams.split('-')
    beams = np.array(range(int(b_range[1])-int(b_range[0])+1)) + int(b_range[0])
else:
    beams = [int(b) for b in args.beams.split(',')]

overwrite = args.overwrite

cube_name = 'HI_image_cube'
beam_name = 'HI_beam_cube'
alta_dir = '/altaZone/archive/apertif_main/visibilities_default/'

header = ['name', 'id', 'x', 'y', 'z', 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max', 'n_pix',
          'f_min', 'f_max', 'f_sum', 'rel', 'flag', 'rms', 'w20', 'w50', 'ell_maj', 'ell_min', 'ell_pa',
          'ell3s_maj', 'ell3s_min', 'ell3s_pa', 'kin_pa', "err_x", "err_y", "err_z", "err_f_sum", 'taskid', 'beam', 'cube']

catParNames = ("name", "id", "x", "y", "z", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "n_pix",
               "f_min", "f_max", "f_sum", "rel", "flag", "rms", "w20", "w50", "ell_maj", "ell_min", "ell_pa",
               "ell3s_maj", "ell3s_min", "ell3s_pa", "kin_pa", "err_x", "err_y", "err_z", "err_f_sum", "taskid", "beam", "cube")
catParUnits = ("-", "-", "pix", "pix", "chan", "pix", "pix", "pix", "pix", "chan", "chan", "-",
               "Jy/beam", "Jy/beam", "Jy/beam", "-", "-", "Jy/beam", "chan", "chan", "pix", "pix", "pix",
               "pix", "pix", "deg", "deg", "pix", "pix", "pix", "Jy/beam", "-", "-", "-")
catParFormt = ("%12s", "%7i", "%10.3f", "%10.3f", "%10.3f", "%7i", "%7i", "%7i", "%7i", "%7i", "%7i", "%8i",
               "%10.7f", "%10.7f", "%12.6f", "%8.6f", "%7i", "%12.6f", "%10.3f", "%10.3f", "%10.3f", "%10.3f", "%10.3f",
               "%10.3f", "%10.3f", "%10.3f", "%10.3f", "%10.3f", "%10.3f", "%10.3f", "%12.6f", "%10i", "%7i", "%7i")
prepare = apercal.prepare()

for b in beams:
    loc = '/tank/hess/apertif/' + taskid + '/B0' + str(b).zfill(2) + '/'
    print("\t{}".format(loc))
    clean_catalog = loc + 'clean_cat.txt'

    managefiles.director(prepare, 'ch', loc)

    for c in cubes:
        line_cube = cube_name + '{0}.fits'.format(c)
        beam_cube = beam_name + '{0}.fits'.format(c)
        maskfits = cube_name + '{0}_4sig_mask.fits'.format(c)
        mask2dfits = cube_name + '{0}_4sig_mask-2d.fits'.format(c)
        filteredfits = cube_name + '{0}_filtered.fits'.format(c)
        splinefits = cube_name + '{0}_filtered_spline.fits'.format(c)
        new_splinefits = cube_name + '{0}_all_spline.fits'.format(c)
        catalog_file = cube_name + '{0}_4sig_cat.txt'.format(c)

        if os.path.isfile(maskfits):
            catalog = ascii.read(catalog_file, header_start=10)
            if args.sources == 'all':
                mask_expr = '"(<mask_sofia>.eq.-1).or.(<mask_sofia>.ge.1)"'
                sources = [str(s + 1) for s in range(len(catalog))]
            elif '-' in args.sources:
                mask_range = args.sources.split('-')
                sources = [str(s + int(mask_range[0])) for s in range(int(mask_range[1]) - int(mask_range[0]) + 1)]
                mask_expr = '"(<mask_sofia>.eq.-1).or.((<mask_sofia>.ge.{}).and.(<mask_sofia>.le.{}))"'.format(
                    mask_range[0],
                    mask_range[-1])
            else:
                sources = [str(s) for s in args.sources.split(',')]
                mask_expr = '"(<mask_sofia>.eq.-1).or.(<mask_sofia>.eq.' + ').or.(<mask_sofia>.eq.'.join(sources) + ')"'

            # If cleaning the filtered_spline cube, rather than original data: do some repair work.
            if (not args.nospline) & ((not os.path.isfile(new_splinefits)) | args.overwrite):
                print("[CLEAN] Creating a 'repaired' spline cube for Beam {:02}, Cube {}.".format(b, c))
                os.system('cp {} {}'.format(splinefits, new_splinefits))
                print("\t{}".format(new_splinefits))
                new_splinecube = pyfits.open(new_splinefits, mode='update')
                # maskcube = pyfits.open(maskfits)  # For multiple source along line of sight...need to develop!
                mask2d = pyfits.getdata(mask2dfits)
                orig_data = pyfits.getdata(line_cube)
                new_splinecube_data = new_splinecube[0].data
                filtered_pixels = np.copy(new_splinecube[0].data[0, :, :])

                ################################################
                # Parallelization of repair
                xx, yy = range(filtered_pixels.shape[0]), range(filtered_pixels.shape[1])
                x, y = np.meshgrid(xx, yy)
                x, y = x.ravel(), y.ravel()
                ncases = len(x)
                print(" - " + str(ncases) + " cases found")

                if njobs > 1:
                    print(" - Running in parallel mode (" + str(njobs) + " jobs simultaneously)")
                elif njobs == 1:
                    print(" - Running in serial mode")
                else:
                    print("[ERROR] invalid number of NJOBS. Please use a positive number.")
                    exit()

                # Managing the work PARALLEL or SERIAL accordingly
                if njobs > cpu_count():
                    print(
                        "  [WARNING] The chosen number of NJOBS seems to be larger than the number of CPUs in the system!")

                # Create Queues
                print("    - Creating Queues")
                inQueue = Queue()
                outQueue = Queue()

                # Create worker processes
                print("    - Creating worker processes")
                ps = [Process(target=worker, args=(inQueue, outQueue)) for _ in range(njobs)]

                # Start worker processes
                print("    - Starting worker processes")
                for p in ps: p.start()

                # Fill the queue
                print("    - Filling up the queue")
                for i in trange(ncases):
                    inQueue.put((i))

                # Now running the processes
                print("    - Running the processes")
                output = [outQueue.get() for _ in trange(ncases)]

                # Send stop signal to stop iteration
                for _ in range(njobs): inQueue.put('STOP')

                # Stop processes
                print("    - Stopping processes")
                for p in ps: p.join()

                # Updating the Splinecube file with the new data
                print(" - Updating the Splinecube file")
                new_splinecube.data = new_splinecube_data
                new_splinecube.flush()

                # Closing files
                print(" - Closing files")
                new_splinecube.close()

                ################################################

            if args.nospline:
                f = pyfits.open(filteredfits)
                print("[CLEAN] Determining the statistics from the filtered Beam {:02}, Cube {}.".format(b, c))
            else:
                if os.path.isfile(splinefits):
                    f = pyfits.open(splinefits)
                else:
                    f = pyfits.open(new_splinefits)
                print("[CLEAN] Determining the statistics from the filtered & spline fitted Beam {:02}, Cube {}.".format(b, c))
            mask = np.ones(f[0].data.shape[0], dtype=bool)
            if c == 3: mask[376:662] = False
            lineimagestats = [np.nanmin(f[0].data[mask]), np.nanmax(f[0].data[mask]), np.nanstd(f[0].data[mask])]
            f.close()
            print("\tImage min, max, std: {}".format(lineimagestats[:]))

            # Output what exactly is being used to clean the data
            print("\t{}".format(maskfits))
            # Edit mask cube to trick Miriad into using the whole volume.
            m = pyfits.open(maskfits, mode='update')
            m[0].data[0, 0, 0] = -1
            m[0].data[-1, -1, -1] = -1
            m[0].scale('int16')
            m.flush()

            # Delete any pre-existing Miriad files.
            os.system('rm -rf model_* beam_* map_* image_* mask_* residual_*')

            print("[CLEAN] Reading in FITS files, making Miriad mask.")

            fits = lib.miriad('fits')
            fits.op = 'xyin'
            if args.nospline:
                fits.in_ = line_cube
            else:
                fits.in_ = new_splinefits
            fits.out = 'map_00'
            fits.go()

            if not os.path.isfile(beam_cube):
                print("[CLEAN] Retrieving synthesized beam cube from ALTA.")
                os.system('iget {}{}_AP_B0{:02}/HI_beam_cube{}.fits {}'.format(alta_dir, taskid, b, c, loc))
            fits.in_ = beam_cube
            fits.out = 'beam_00'
            fits.go()

            # Work with mask_sofia in current directory...otherwise worried about miriad character length for mask_expr
            fits.in_ = maskfits
            fits.out = 'mask_sofia'
            fits.go()

            maths = lib.miriad('maths')
            maths.out = 'mask_00'
            maths.exp = '"<mask_sofia>"'
            maths.mask = mask_expr
            maths.go()

            nminiter = 1
            for minc in range(nminiter):
                print("[CLEAN] Cleaning HI emission using SoFiA mask for Sources {}.".format(args.sources))
                clean = lib.miriad('clean')
                clean.map = 'map_' + str(minc).zfill(2)
                clean.beam = 'beam_' + str(minc).zfill(2)
                clean.out = 'model_' + str(minc + 1).zfill(2)
                clean.cutoff = lineimagestats[2] * 0.5
                clean.region = '"' + 'mask(mask_' + str(minc).zfill(2) + '/)"'
                clean.go()

                print("[CLEAN] Restoring line cube.")
                restor = lib.miriad('restor')  # Create the restored image
                restor.model = 'model_' + str(minc + 1).zfill(2)
                restor.beam = 'beam_' + str(minc).zfill(2)
                restor.map = 'map_' + str(minc).zfill(2)
                restor.out = 'image_' + str(minc + 1).zfill(2)
                restor.mode = 'clean'
                restor.go()

                # print("[CLEAN] Making residual cube.")
                # restor.mode = 'residual'  # Create the residual image
                # restor.out = loc + 'residual_' + str(minc + 1).zfill(2)
                # restor.go()

            if overwrite:
                os.system('rm {}_clean.fits {}_residual.fits {}_model.fits'.format(line_cube[:-5], line_cube[:-5],
                                                                                   line_cube[:-5]))
                print("\tWARNING...overwrite won't delete clean_cat.txt file.  Manage this carefully!")

            print("[CLEAN] Writing out cleaned image, residual, and model to FITS.")
            fits.op = 'xyout'
            fits.in_ = 'image_' + str(minc + 1).zfill(2)
            if args.nospline:
                fits.out = line_cube[:-5] + '_clean.fits'
            else:
                fits.out = line_cube[:-5] + '_rep_clean.fits'
            fits.go()

            # fits.op = 'xyout'
            # fits.in_ = loc + 'residual_' + str(minc + 1).zfill(2)
            # fits.out = line_cube[:-5] + '_residual.fits'
            # fits.go()

            fits.in_ = loc + 'model_' + str(minc + 1).zfill(2)
            if args.nospline:
                fits.out = line_cube[:-5] + '_model.fits'
            else:
                fits.out = line_cube[:-5] + '_rep_model.fits'
            fits.go()

            catalog = ascii.read(catalog_file, header_start=10)
            catalog['taskid'] = np.int(taskid.replace('/', ''))
            catalog['beam'] = b
            catalog['cube'] = c
            # Not sure that I've actually reordered anything.  This might be hold over that I gave up on:
            catalog_reorder = catalog['name', 'id', 'x', 'y', 'z', 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max',
                                      'n_pix', 'f_min', 'f_max', 'f_sum', 'rel', 'flag', 'rms', 'w20', 'w50',
                                      'ell_maj', 'ell_min', 'ell_pa', 'ell3s_maj', 'ell3s_min', 'ell3s_pa', 'kin_pa',
                                      "err_x", "err_y", "err_z", "err_f_sum", 'taskid', 'beam', 'cube']

            if args.sources == 'all':
                sources = [str(s+1) for s in range(len(catalog))]

            # If everything was successful and didn't crash for a given beam/cube:
            # Copy SoFiA catalog for *cleaned* sources to clean_cat.txt (Same for all cubes in a beam).
            objects = []
            for source in catalog_reorder:
                if str(source['id']) in sources:
                    obj = []
                    for s in source:
                        obj.append(s)
                    objects.append(obj)

            if args.nospline:
                print("[CLEAN] Writing/updating cleaned source catalog: clean_cat.txt")
                write_catalog(objects, catParNames, catParUnits, catParFormt, header, outName=loc+'clean_cat.txt')
            else:
                print("[CLEAN] Writing/updating cleaned source catalog: rep_clean_cat.txt")
                write_catalog(objects, catParNames, catParUnits, catParFormt, header, outName=loc + 'rep_clean_cat.txt')

            # Clean up extra Miriad files
            os.system('rm -rf model_* beam_* map_* image_* mask_* residual_*')

    # Will probably need to do some sorting of the catalog if run clean multiple times.  This is a starting point:
    # os.system('head -n +1 {} > temp'.format(clean_catalog))
    # os.system('tail -n +2 {} | sort | uniq > temp2'.format(clean_catalog))

print("[CLEAN] Done.")
