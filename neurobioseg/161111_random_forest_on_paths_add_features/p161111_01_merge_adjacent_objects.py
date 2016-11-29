
import os
import inspect
from hdf5_image_processing import Hdf5ImageProcessing as IP, Hdf5ImageProcessingLib as IPL
from hdf5_processing import RecursiveDict as rdict
from shutil import copy, copyfile
import numpy as np
import matplotlib.pyplot as plt
import processing_libip as libip
import sys


__author__ = 'jhennies'

def load_images(ipl, path, filename, dataname):

    ipl.data_from_file(
        filepath=path + filename, skeys=dataname,
        recursive_search=True, nodata=True
    )


def merge_adjacent_objects(ipl):

    params = ipl.get_params()
    thisparams = rdict(params['merge_adjacent_objects'])
    targetfile = params['intermedfolder'] + params['largeobjmfile']

    # Load the necessary images
    load_images(ipl, params['intermedfolder'], params['largeobjfile'], params['largeobjname'])
    ipl.logging('\nInitial datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

    for d, k, v, kl in ipl.data_iterator(yield_short_kl=True):

        if k == params['largeobjname']:

            ipl.logging('===============================\nWorking on image: {}', kl + [k])

            ipl[kl].set_logger(ipl.get_logger())

            # Load the image data into memory
            ipl[kl].populate(k)

            ipl[kl] = libip.merge_adjacent_objects(
                ipl[kl], k,
                thisparams['numberbysize'], thisparams['numberbyrandom'], thisparams['seed'],
                targetnames=params['largeobjmnames'], algorithm=thisparams['algorithm']
            )

            # Write the result to file
            ipl.write(filepath=targetfile, keys=[kl])
            # Free memory
            ipl[kl] = None



def run_merge_adjacent_objects(yamlfile):

    ipl = IPL(yaml=yamlfile)

    ipl.set_indent(1)

    params = rdict(data=ipl.get_params())
    ipl.startlogger(filename=params['resultfolder'] + 'merge_adjacent_objects.log', type='w', name='MergeAdjacentObjects')

    try:

        # # Copy the script file and the parameters to the scriptsfolder
        # copy(inspect.stack()[0][1], params['scriptsfolder'])
        # copy(yamlfile, params['scriptsfolder'] + 'merge_adjacent_objects.parameters.yml')

        # ipl.logging('\nInitial datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

        merge_adjacent_objects(ipl)

        # ipl.logging('\nFinal datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

        # ipl.write(filepath=params['intermedfolder'] + params['largeobjfile'])

        ipl.logging('')
        ipl.stoplogger()

    except:

        ipl.errout('Unexpected error')


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'

    run_merge_adjacent_objects(yamlfile)