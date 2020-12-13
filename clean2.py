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
            temp = np.copy(orig_data[:, x[i], y[i]])
            fit = fspline(np.linspace(1, orig_data.shape[0], orig_data.shape[0]), np.nan_to_num(temp), k=5)
            new_splinecube_data[:, x[i], y[i]] = temp - fit
        # Second, use source mask to undo potential over subtraction:
        if str(mask2d[x[i], y[i]]) in sources:
            s = mask2d[x[i], y[i]]
            zmin, zmax = np.int(catalog[catalog['id'] == s]['z_min']), np.int(catalog[catalog['id'] == s]['z_max'])
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


def worker2(inQueue, outQueue):

    """
    Defines the worker process of the parallelisation with multiprocessing.Queue
    and multiprocessing.Process.
    """
    for i in iter(inQueue.get, 'STOP'):

        status = run2(i)

        outQueue.put(( status ))


def run2(i):
    global new_cleancube_data, new_residualcube_data
    # global new_cleancube_data, new_modelcube_data, new_residualcube_data

    # try:
    for name in ['map_{:02}'.format(minc), 'beam_{:02}'.format(minc), 'mask_{:02}'.format(minc)]:
        imsub = lib.miriad('imsub')
        imsub.in_ = name
        imsub.out = name + "_" + str(chan[i]).zfill(4)
        imsub.region = '"images({})"'.format(chan[i] + 1)
        imsub.go()

    # print("[CLEAN] Cleaning HI emission using SoFiA mask for Sources {}.".format(args.sources))
    clean.map = 'map_{:02}_{:04}'.format(minc, chan[i])
    clean.beam = 'beam_{:02}_{:04}'.format(minc, chan[i])
    clean.out = 'model_{:02}_{:04}'.format(minc + 1, chan[i])
    clean.cutoff = lineimagestats[2] * 0.5
    clean.region = '"mask(mask_{:02}_{:04}/)"'.format(minc, chan[i])
    clean.go()

    # print("[CLEAN] Restoring line cube.")
    restor.model = 'model_{:02}_{:04}'.format(minc + 1, chan[i])
    restor.beam = 'beam_{:02}_{:04}'.format(minc, chan[i])
    restor.map = 'map_{:02}_{:04}'.format(minc, chan[i])
    restor.out = 'image_{:02}_{:04}'.format(minc + 1, chan[i])
    restor.mode = 'clean'
    restor.go()

    # print("[CLEAN] Making residual cube.")
    restor.mode = 'residual'  # Create the residual image
    restor.out = loc + 'residual_{:02}_{:04}'.format(minc + 1, chan[i])
    restor.go()

    for name in ['model_{:02}_{:04}'.format(minc + 1, chan[i]), 'image_{:02}_{:04}'.format(minc + 1, chan[i]),
             'residual_{:02}_{:04}'.format(minc + 1, chan[i])]:
        fits.op = 'xyout'
        fits.in_ = name
        fits.out = name + '.fits'
        fits.go()

    new_cleancube_data[chan[i], :, :] = pyfits.getdata('image_{:02}_{:04}.fits'.format(minc + 1, chan[i]))
    # new_modelcube_data[chan[i], :, :] = pyfits.getdata('model_{:02}_{:04}.fits'.format(minc + 1, chan[i]))
    new_residualcube_data[chan[i], :, :] = pyfits.getdata('residual_{:02}_{:04}.fits'.format(minc + 1, chan[i]))

    return 'OK'

    # except Exception:
    #     print("[ERROR] Something went wrong with the cleaning! [Channel {}]".format(chan[i]))
    #     return np.nan

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
                print("[CLEAN2] Creating a 'repaired' spline cube for Beam {:02}, Cube {}.".format(b, c))
                os.system('cp {} {}'.format(splinefits, new_splinefits))  #speed things up by just writing later rather than copying?
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
            elif os.path.isfile(splinefits):
                f = pyfits.open(splinefits)
                print("[CLEAN] Determining the statistics from the filtered & spline fitted Beam {:02}, Cube {}.".format(b, c))
            else:
                f = pyfits.open(new_splinefits)
                print("[CLEAN] Determining the statistics from the repaired Beam {:02}, Cube {}.".format(b, c))
            nchan = f[0].data.shape[0]
            mask = np.ones(nchan, dtype=bool)
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
            m.close()

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

            print("[CLEAN] Initialize clean, model, and residual cubes")
            if args.nospline:
                dirty_cube = line_cube
                outcube = line_cube[:-5]
            else:
                dirty_cube = new_splinefits
                outcube = line_cube[:-5] + '_rep'

            new_cleanfits = outcube + '_clean.fits'
            os.system('cp {} {}'.format(dirty_cube, new_cleanfits))
            print("\t{}".format(new_cleanfits))
            new_cleancube = pyfits.open(new_cleanfits, mode='update')
            new_cleancube_data = new_cleancube[0].data

            # new_modelfits = outcube + '_clean.fits'
            # os.system('cp {} {}'.format(dirty_cube, new_modelfits))
            # print("\t{}".format(new_modelfits))
            # new_modelcube = pyfits.open(new_modelfits, mode='update')
            # new_modelcube_data = new_modelcube[0].data * np.nan

            new_residualfits = outcube + '_residual.fits'
            os.system('cp {} {}'.format(dirty_cube, new_residualfits))
            print("\t{}".format(new_residualfits))
            new_residualcube = pyfits.open(new_residualfits, mode='update')
            new_residualcube_data = new_residualcube[0].data

            # Get channels to clean explicitly!
            chan = []
            for s in sources:
                zmin, zmax = np.int(catalog[catalog['id'] == np.int(s)]['z_min']), \
                             np.int(catalog[catalog['id'] == np.int(s)]['z_max'])
                chan = chan + [c for c in range(zmin, zmax+1, 1)]
            chan = set(chan)
            chan = list(chan)
            bmaj_arr = np.zeros(len(chan))
            bmin_arr = np.zeros(len(chan))
            bpa_arr = np.zeros(len(chan))

            nminiter = 1
            clean = lib.miriad('clean')
            restor = lib.miriad('restor')  # Create the restored image
            for minc in range(nminiter):
                print("[CLEAN2] Clean & restor HI emission using SoFiA mask for Sources {}.".format(args.sources))

                ################################################
                # Parallelization of cleaning/restoring
                ncases = len(chan)
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

                # Create worker2 processes
                print("    - Creating worker2 processes")
                ps = [Process(target=worker2, args=(inQueue, outQueue)) for _ in range(njobs)]

                # Start worker2 processes
                print("    - Starting worker2 processes")
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

                ################################################

            print(" - Updating the clean file")
            new_cleancube.data = new_cleancube_data

            # print(" - Updating the model file")
            # new_modelcube.data = new_modelcube_data
            # new_modelcube.flush()
            # new_modelcube.close()

            print(" - Updating the residual file")
            new_residualcube.data = new_residualcube_data

            print("[CLEAN2] Updating history of reassembled clean,[model,]residual cubes")
            clean_chan_hdr = pyfits.getheader('image_{:02}_{:04}.fits'.format(minc + 1, chan[0]))
            residual_chan_hdr = pyfits.getheader('residual_{:02}_{:04}.fits'.format(minc + 1, chan[0]))
            for hist in clean_chan_hdr[-41:]['HISTORY']:  # Determined through trial and error
                new_cleancube[0].header['HISTORY'] = hist
            for hist in residual_chan_hdr[-41:]['HISTORY']:
                new_residualcube[0].header['HISTORY'] = hist
            bmaj_arr[0] = clean_chan_hdr['BMAJ']
            bmin_arr[0] = clean_chan_hdr['BMIN']
            bpa_arr[0] = clean_chan_hdr['BPA']
            for i in range(1,len(chan)):
                clean_chan_hdr = pyfits.getheader('image_{:02}_{:04}.fits'.format(minc + 1, chan[i]))
                residual_chan_hdr = pyfits.getheader('residual_{:02}_{:04}.fits'.format(minc + 1, chan[i]))
                for hist in clean_chan_hdr[-35:]['HISTORY']:  # Determined through trial and error
                    new_cleancube[0].header['HISTORY'] = hist
                for hist in residual_chan_hdr[-35:]['HISTORY']:
                    new_residualcube[0].header['HISTORY'] = hist
                bmaj_arr[i] = clean_chan_hdr['BMAJ']
                bmin_arr[i] = clean_chan_hdr['BMIN']
                bpa_arr[i] = clean_chan_hdr['BPA']
            new_cleancube[0].header['HISTORY'] = 'Individual images reassembled using sourcefinding/clean2.py by KMHess'

            print("[CLEAN2] Adding median beam properties to primary header")
            med_bmaj, med_bmin, med_bpa = np.median(bmaj_arr), np.median(bmin_arr), np.median(bpa_arr)
            new_cleancube[0].header.set('BMAJ', med_bmaj, 'median clean beam bmaj')
            new_cleancube[0].header.set('BMIN', med_bmin, 'median clean beam bmin')
            new_cleancube[0].header.set('BPA', med_bpa, 'median clean beam pa')

            print("[CLEAN2] Adding channel clean beam properties to BEAMS extension table")
            col1 = pyfits.Column(name='BMAJ', format='1E', unit='deg', array=bmaj_arr)
            col2 = pyfits.Column(name='BMIN', format='1E', unit='deg', array=bmin_arr)
            col3 = pyfits.Column(name='BPA', format='1E', unit='deg', array=bpa_arr)
            col4 = pyfits.Column(name='CHAN', format='1J', array=chan)
            beam_hdu = pyfits.BinTableHDU.from_columns([col1, col2, col3, col4])
            beam_hdu.name = 'BEAMS'
            beam_hdu.header.comments['NAXIS2'] = 'number of channels'
            new_cleancube.append(beam_hdu)

            print(" - Saving the new clean file")
            new_cleancube.flush()
            new_cleancube.close()

            print(" - Saving the new residual file")
            new_residualcube.flush()
            new_residualcube.close()

            # Output sofia only mask????
            print("[CLEAN2] Writing mask with only cleaned sources")
            os.system('rm -rf {}_clean_mask.fits'.format(outcube))
            fits.op = 'xyout'
            fits.in_ = 'mask_' + str(minc).zfill(2)
            if not args.nospline:
                fits.out = outcube + '_clean_mask.fits'
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

            # If everything was successful and didn't crash for a given beam/cube:
            # Copy SoFiA catalog for *cleaned* sources to [rep_]clean_cat.txt (Same for all cubes in a beam).
            objects = []
            for source in catalog_reorder:
                if str(source['id']) in sources:
                    obj = []
                    for s in source:
                        obj.append(s)
                    objects.append(obj)

            if args.nospline:
                print("[CLEAN2] Writing/updating cleaned source catalog: clean_cat.txt")
                write_catalog(objects, catParNames, catParUnits, catParFormt, header, outName=loc+'clean_cat.txt')
            else:
                print("[CLEAN2] Writing/updating cleaned source catalog: rep_clean_cat.txt")
                write_catalog(objects, catParNames, catParUnits, catParFormt, header, outName=loc + 'rep_clean_cat.txt')

            # Clean up extra Miriad files
            os.system('rm -rf model_* beam_* map_* image_* mask_* residual_*')

    # Will probably need to do some sorting of the catalog if run clean multiple times.  This is a starting point:
    # os.system('head -n +1 {} > temp'.format(clean_catalog))
    # os.system('tail -n +2 {} | sort | uniq > temp2'.format(clean_catalog))

print("[CLEAN2] Done.")
