
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
    paths_true (paths within single label objects)
    paths_false (paths of merged objects which cross the merging site)
    featureims_true
    featureims_false
    :param ipl:
    :return:
    """
    paths_true = IPL()
    paths_false = IPL()
    featureims_true = IPL()
    featureims_false = IPL()

    params = ipl.get_params()

    # Paths within labels (true paths)
    paths_true.data_from_file(
        filepath=params['intermedfolder'] + params['pathstruefile'],
        skeys='path',
        recursive_search=True, nodata=True
    )

    # Paths of merges (false paths)
    paths_false.data_from_file(
        filepath=params['intermedfolder'] + params['pathsfalsefile'],
        skeys='path',
        recursive_search=True, nodata=True
    )

    # Load features for true paths
    featureims_true.data_from_file(
        filepath=params['intermedfolder'] + params['featureimsfile'],
        nodata=True
    )
    featureims_true.delete_items(params['largeobjmnames'][0])

    # Load features for false paths
    featureims_false.data_from_file(
        filepath=params['intermedfolder'] + params['featureimsfile'],
        nodata=True
    )
    featureims_false.delete_items(params['largeobjname'])

    return (paths_true, paths_false, featureims_true, featureims_false)

def features_of_paths(ipl):

    params = ipl.get_params()
    thisparams = rdict(params['features_of_paths'])
    targetfile = params['intermedfolder'] + params['featuresfile']

    # Load the necessary images
    paths_true, paths_false, featureims_true, featureims_false = load_images(ipl)
    ipl.logging('\nInitial datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

    for d, k, v, kl in paths_true.data_iterator(yield_short_kl=True):

        if k == 'path':

            ipl.logging('===============================\nWorking on group: {}', kl)

            # # TODO: Implement copy full logger
            # ipl[kl].set_logger(ipl.get_logger())

            # # Load the image data into memory
            # ipl[kl].populate()

            ipl[kl] = libip.features_of_paths(
                ipl,
                paths_true[kl][k], paths_false[kl][k],
                featureims_true[kl], featureims_false[kl],
                kl
            )

            # Write the result to file
            ipl.write(filepath=targetfile, keys=[kl])
            # # Free memory
            ipl[kl] = None



def run_features_of_paths(yamlfile):

    ipl = IPL(yaml=yamlfile)

    ipl.set_indent(1)

    params = rdict(data=ipl.get_params())
    ipl.startlogger(filename=params['resultfolder'] + 'features_of_paths.log', type='w', name='FeaturesOfPaths')

    try:

        # # Copy the script file and the parameters to the scriptsfolder
        # copy(inspect.stack()[0][1], params['scriptsfolder'])
        # copy(yamlfile, params['scriptsfolder'] + 'features_of_paths.parameters.yml')

        # ipl.logging('\nInitial datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

        features_of_paths(ipl)

        # ipl.logging('\nFinal datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

        # features.write(filepath=params['intermedfolder'] + params['featuresfile'])

        ipl.logging('')
        ipl.stoplogger()

    except:
        raise
        ipl.errout('Unexpected error')


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'

    run_features_of_paths(yamlfile)