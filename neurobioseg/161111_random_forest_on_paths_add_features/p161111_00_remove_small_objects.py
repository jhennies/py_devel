
import os
import inspect
from hdf5_image_processing import Hdf5ImageProcessing as IP, Hdf5ImageProcessingLib as IPL
from shutil import copy, copyfile
import numpy as np
import matplotlib.pyplot as plt
import processing_libip as libip
import sys


__author__ = 'jhennies'


def remove_small_objects(ipl):
    """
    :param ipl: A Hdf5ImageProcessingLib instance containing labelimages named as specified in ipl.get_params()['labelsname']

    ipl.get_params()

        remove_small_objects
            bysize
            relabel

        largeobjname

        labelsname

    :param key: the source key for calculation
    """

    params = ipl.get_params()
    thisparams = params['remove_small_objects']
    targetfile = params['intermedfolder'] + params['largeobjfile']

    for d, k, v, kl in ipl.data_iterator(yield_short_kl=True):

        if k == params['labelsname']:

            ipl.logging('===============================\nWorking on image: {}', kl + [k])

            # TODO: Implement copy full logger
            ipl[kl].set_logger(ipl.get_logger())

            # Load the image data into memory
            ipl[kl].populate(k)

            ipl[kl] = libip.filter_small_objects(ipl[kl], k, thisparams['bysize'], relabel=thisparams['relabel'])

            # Rename the entry
            ipl[kl].rename_entry(params['labelsname'], params['largeobjname'])

            # Write the result to file
            ipl.write(filepath=targetfile, keys=[kl + [params['largeobjname']]])
            # Free memory (With this command the original reference to the source file is restored)
            ipl[kl].unpopulate()


def run_remove_small_objects(yamlfile):

    ipl = IPL(
        yaml=yamlfile,
        yamlspec={'path': 'datafolder', 'filename': 'labelsfile', 'skeys': 'labelsname'},
        recursive_search=True,
        nodata=True
    )

    # Set indentation of the logging
    ipl.set_indent(1)

    params = ipl.get_params()
    ipl.startlogger(filename=params['resultfolder'] + 'remove_small_objects.log', type='w', name='RemoveSmallObjects')

    try:

        # Copy the script file and the parameters to the scriptsfolder
        copy(inspect.stack()[0][1], params['scriptsfolder'])
        copy(yamlfile, params['scriptsfolder'] + 'remove_small_objects.parameters.yml')

        ipl.logging('\nipl datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

        remove_small_objects(ipl)

        ipl.logging('\nFinal datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

        # ipl.write(filepath=params['intermedfolder'] + params['largeobjfile'])

        ipl.logging('')
        ipl.stoplogger()

    except:

        ipl.errout('Unexpected error')


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'

    run_remove_small_objects(yamlfile)

