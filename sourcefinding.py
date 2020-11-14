import os

from modules.natural_cubic_spline import fspline
from src import checkmasks

from argparse import ArgumentParser, RawTextHelpFormatter
from astropy.io import fits
import numpy as np

from multiprocessing import Queue, Process, cpu_count
from tqdm.auto import trange


def make_param_file(sig=4, loc_dir=None, cube_name=None, cube=None):
    param_template = 'parameter_template_{}sig.par'.format(sig)
    new_paramfile = loc_dir + 'parameter_{}sig.par'.format(sig)
    outlog = loc_dir + 'sourcefinding_{}sig.out'.format(sig)

    # Edit parameter file (remove lines that need editing)
    os.system('grep -vwE "(input.data)" ' + param_template + ' > ' + new_paramfile)
    os.system('grep -vwE "(output.filename)" ' + new_paramfile + ' > temp && mv temp ' + new_paramfile)
    if cube == 3:
        os.system('grep -vwE "(flag.region)" ' + new_paramfile + ' > temp && mv temp ' + new_paramfile)
        os.system('grep -vwE "(linker.maxSizeXY)" ' + new_paramfile + ' > temp && mv temp ' + new_paramfile)
        os.system('grep -vwE "(linker.maxSizeZ)" ' + new_paramfile + ' > temp && mv temp ' + new_paramfile)

    # Add back the parameters needed
    if not args.nospline:
        os.system('echo "input.data                 =  ' + splinefits + '" >> ' + new_paramfile)
        outroot = cube_name + '_{}sig'.format(sig)
    else:
        os.system('echo "input.data                 =  ' + filteredfits + '" >> ' + new_paramfile)
        outroot = cube_name + '_{}sig'.format(sig)

    os.system('echo "output.filename            =  ' + outroot + '" >> ' + new_paramfile)
    if cube == 3:
        os.system('echo "flag.region                =  0,661,0,661,375,601" >> ' + new_paramfile)
        os.system('echo "linker.maxSizeXY           =  250" >> ' + new_paramfile)
        os.system('echo "linker.maxSizeZ            =  385" >> ' + new_paramfile)

    return new_paramfile, outlog

def worker(inQueue, outQueue):

    """
    Defines the worker process of the parallelisation with multiprocessing.Queue
    and multiprocessing.Process.
    """
    for i in iter(inQueue.get, 'STOP'):

        status = run(i)

        outQueue.put(( status ))


def run(i):
    global splinecube_data

    try:
        # Do the spline fitting on the z-axis to masked cube
        fit = fspline(np.linspace(1, orig_data.shape[0], orig_data.shape[0]),
                      np.nan_to_num(splinecube_data[:, x[i], y[i]]), k=5)
        splinecube_data[:, x[i], y[i]] = orig_data[:, x[i], y[i]] - fit
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

parser.add_argument('-o', "--overwrite",
                    help="If option is included, overwrite old continuum filtered and/or spline fitted file if either exists.",
                    action='store_true')

parser.add_argument('-n', "--nospline",
                    help="Don't do spline fitting; so source finding on only continuum filtered cube.",
                    action='store_true')

parser.add_argument('-j', "--njobs",
                    help="Number of jobs to run in parallel (default: %(default)s) tested on happili-05.",
                    default=18)

# Parse the arguments above
args = parser.parse_args()
njobs = int(args.njobs)

# Range of cubes/beams to work on:
taskid = args.taskid
cubes = [int(c) for c in args.cubes.split(',')]
if '-' in args.beams:
    b_range = args.beams.split('-')
    beams = np.array(range(int(b_range[1]) - int(b_range[0]) + 1)) + int(b_range[0])
else:
    beams = [int(b) for b in args.beams.split(',')]
overwrite = args.overwrite

# Main source finding code for all cubes/beams
for b in beams:
    # Define some file names and work space:
    loc = '/tank/hess/apertif/' + taskid + '/B0' + str(b).zfill(2) + '/'

    for c in cubes:
        cube_name = 'HI_image_cube' + str(c)
        print("[SOURCEFINDING] Working on Beam {:02} Cube {}".format(b, c))

        sourcefits = loc + cube_name + '.fits'
        filteredfits = loc + cube_name + '_filtered.fits'
        splinefits = loc + cube_name + '_filtered_spline.fits'
        # Output exactly where sourcefinding is starting
        print('\t' + sourcefits)

###############################################
# Can this be parallelized? Especially the two for loops.

        # Check to see if the continuum filtered file exists.  If not, make it  with SoFiA-2
        if (not overwrite) & os.path.isfile(filteredfits):
            print("[SOURCEFINDING] Continuum filtered file exists and will not be overwritten.")
        elif os.path.isfile(sourcefits):
            print("[SOURCEFINDING] Making continuum filtered file.")
            os.system('grep -vwE "(input.data)" template_filtering.par > ' + loc + 'filtering.par')
            os.system('echo "input.data                 =  ' + sourcefits + '" >> ' + loc + 'filtering.par')
            os.system('/home/apercal/SoFiA-2/sofia ' + loc + 'filtering.par >> test.log')
        else:
            print("\tBeam {:02} Cube {} is not present in this directory.".format(b, c))
            continue

        # Check to see if the spline fitted file exists.  If not, make it from filtered file.
        if (not overwrite) & os.path.isfile(splinefits):
            print("[SOURCEFINDING] Spline fitted file exists and will not be overwritten.")
        elif os.path.isfile(filteredfits) & (not args.nospline):
            print(" - Loading the input cube")
            os.system('cp {} {}'.format(filteredfits, splinefits))
            splinecube = fits.open(splinefits, mode='update')
            orig = fits.open(filteredfits)
            orig_data = orig[0].data
            splinecube_data = splinecube[0].data

            # Try masking strong sources to not bias fit
            print(" - Masking strong sources to not bias fit")
            mask = 2.5 * np.nanstd(orig_data)
            splinecube_data[np.abs(splinecube_data) >= mask] = np.nan

            # Defining the cases to analyse
            print(" - Defining the cases to analyse")
            xx = range(orig_data.shape[1])
            yy = range(orig_data.shape[2])
            x, y = np.meshgrid(xx, yy)
            x = x.ravel()
            y = y.ravel()
            ncases = len(x)
            # ncases = 10000
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
            splinecube.data = splinecube_data
            splinecube.flush()

            # Closing files
            print(" - Closing files")
            orig.close()
            splinecube.close()

        ################################################
        elif os.path.isfile(sourcefits) & args.nospline:
            print("\tWill not perform spline fitting.  Do source finding on just continuum filtered file.")
            print("\t [WARNING]: this is not the default but the file names are the SAME! Keep track of what you're doing for future steps !!!")
        else:
            print("\tBeam {:02} Cube {} is not present in this directory.".format(b, c))
            continue

        print("[SOURCEFINDING] Doing source finding with 4 sigma threshold.")
        sig = 4
        new_paramfile, outlog = make_param_file(sig=sig, loc_dir=loc, cube_name=cube_name, cube=c)
        os.system('/home/apercal/SoFiA-2/sofia ' + new_paramfile + ' >> ' + outlog)

    # After all cubes are done, run checkmasks to get summary plots for cleaning:
    checkmasks.main(taskid, [b], args.nospline)
