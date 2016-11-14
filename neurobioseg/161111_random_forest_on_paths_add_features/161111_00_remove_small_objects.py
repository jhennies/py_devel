
import os
import inspect
from hdf5_image_processing import Hdf5ImageProcessing as IP, Hdf5ImageProcessingLib as IPL
from shutil import copy, copyfile
import numpy as np
import matplotlib.pyplot as plt
import processing_libip as libip



__author__ = 'jhennies'


def remove_small_objects(ipl):
    """
    :param hfp: A Hdf5ImageProcessing instance containing a labelimage named 'labels'

    hfp.get_params()

        remove_small_objects
            bysize
            relabel

        largeobjname

    :param key: the source key for calculation
    """

    params = ipl.get_params()
    thisparams = params['remove_small_objects']

    for d, k, v, kl in ipl.data_iterator(yield_short_kl=True):

        if k == params['labelsname']:

            ipl.logging('===============================\nWorking on image: {}', kl + [k])

            ipl[kl].setlogger(ipl.getlogger())
            ipl[kl] = libip.filter_small_objects(ipl[kl], k, params, thisparams)


def run_remove_small_objects(resultfolder):

    yamlfile = resultfolder + '/parameters.yml'

    ipl = IPL(
        yaml=yamlfile,
        yamlspec={'path': 'datafolder', 'filename': 'labelsfile'},
        skeys=[['x', '1'], ['x', '0']],
        recursive_search=False,
        nodata=True
    )
    ipl.data_from_file(
        filepath='/mnt/localdata01/jhennies/neuraldata/results/cremi_2016/161110_random_forest_of_paths/intermediate/'
        + 'cremi.splA.raw_neurons.crop.split_xyz.locmaxborder.h5',
        skeys=['disttransf', 'disttransfm'],
        recursive_search=True, nodata=False
    )
    ipl.unpopulate()
    params = ipl.get_params()
    ipl.startlogger(filename=params['resultfolder'] + 'remove_small_objects.log', type='w')

    try:

        # Copy the script file and the parameters to the scriptsfolder
        copy(inspect.stack()[0][1], params['scriptsfolder'])
        copy(yamlfile, params['scriptsfolder'] + 'remove_small_objects.parameters.yml')

        ipl.logging('\nipl datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

        remove_small_objects(ipl)

        ipl.logging('\nFinal datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))


        ipl.write(filepath=params['intermedfolder'] + params['largeobjfile'])

        ipl.logging('')
        ipl.stoplogger()

    except:

        ipl.errout('Unexpected error')


if __name__ == '__main__':

    resultfolder = '/mnt/localdata02/jhennies/neuraldata/results/cremi_2016/161111_random_forest_of_paths_add_features_develop/'

    run_remove_small_objects(resultfolder)

