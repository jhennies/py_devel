
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

def load_images(ipl):
    """
    These images are loaded:
    largeobj (labels)
    largeobjm (merged labels)
    changehash (change hash created by label merging)
    borderct (border contact image)
    disttransf (distance transform of labels)
    :param ipl:
    :return:
    """

    params = ipl.get_params()

    # Load distance transform
    ipl.data_from_file(
        filepath=params['intermedfolder'] + params['featureimsfile'],
        skeys='disttransf',
        recursive_search=True, nodata=True
    )
    ipl.setdata(ipl.subset('raw', search=True))
    ipl.setdata(ipl.subset(params['largeobjname'], search=True))
    ipl.reduce_from_leafs()
    ipl.reduce_from_leafs()
    ipl.rename_entry(params['largeobjname'], 'disttransf', search=True)

    # Load labels
    ipl.data_from_file(
        filepath=params['intermedfolder'] + params['largeobjfile'],
        skeys=params['largeobjname'],
        recursive_search=True, nodata=True
    )

    # Load merged labels and change hash
    ipl.data_from_file(
        filepath=params['intermedfolder'] + params['largeobjmfile'],
        skeys=(params['largeobjmnames'][0], params['largeobjmnames'][4]),
        recursive_search=True, nodata=True
    )

    # Load border contact image
    ipl.data_from_file(
        filepath=params['intermedfolder'] + params['borderctfile'],
        skeys=params['largeobjname'],
        recursive_search=True, nodata=True
    )

    ipl.reduce_from_leafs()


def paths_of_merges(ipl):

    params = ipl.get_params()
    thisparams = rdict(params['paths_of_merges'])
    targetfile = params['intermedfolder'] + params['pathsfalsefile']

    # Load the necessary images
    load_images(ipl)
    ipl.logging('\nInitial datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

    maxd = ipl.maxdepth()
    for d, k, v, kl in ipl.data_iterator(maxdepth=ipl.maxdepth() - 1):

        if d == maxd - 1:

            ipl.logging('===============================\nWorking on group: {}', kl)

            # TODO: Implement copy full logger
            ipl[kl].set_logger(ipl.get_logger())

            # Load the image data into memory
            ipl[kl].populate()

            ipl[kl] = libip.paths_of_labelpairs(
                ipl[kl],
                params['largeobjmnames'][0],
                params['largeobjname'],
                params['largeobjmnames'][4],
                params['borderctname'],
                'disttransf',
                thisparams,
                ignore=thisparams['ignorelabels'],
                max_end_count=thisparams['max_end_count'],
                max_end_count_seed=thisparams['max_end_count_seed']
            )

            # Write the result to file
            ipl.write(filepath=targetfile, keys=[kl])
            # Free memory
            ipl[kl] = None



def run_paths_of_merges(yamlfile):

    ipl = IPL(yaml=yamlfile)

    ipl.set_indent(1)

    params = rdict(data=ipl.get_params())
    ipl.startlogger(filename=params['resultfolder'] + 'paths_of_merges.log', type='w', name='PathsOfMerges')

    try:

        # Copy the script file and the parameters to the scriptsfolder
        copy(inspect.stack()[0][1], params['scriptsfolder'])
        copy(yamlfile, params['scriptsfolder'] + 'paths_of_merges.parameters.yml')

        # ipl.logging('\nInitial datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

        paths_of_merges(ipl)

        # ipl.logging('\nFinal datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

        # ipl.write(filepath=params['intermedfolder'] + params['largeobjfile'])

        ipl.logging('')
        ipl.stoplogger()

    except:

        ipl.errout('Unexpected error')


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'

    run_paths_of_merges(yamlfile)