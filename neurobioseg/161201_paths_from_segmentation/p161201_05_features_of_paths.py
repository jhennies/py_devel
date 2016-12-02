
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
    featureims = IPL()

    params = ipl.get_params()

    ipl.logging('Loading true paths from:\n{} ...', params['intermedfolder'] + params['pathsfile'])
    # Paths within labels (true paths)
    paths_true.data_from_file(
        filepath=params['intermedfolder'] + params['pathsfile'],
        skeys='truepaths',
        recursive_search=True, nodata=True
    )

    ipl.logging('Loading false paths from:\n{} ...', params['intermedfolder'] + params['pathsfile'])
    # Paths of merges (false paths)
    paths_false.data_from_file(
        filepath=params['intermedfolder'] + params['pathsfile'],
        skeys='falsepaths',
        recursive_search=True, nodata=True
    )

    ipl.logging('Loading feature  images from:\n{} ...', params['intermedfolder'] + params['featureimsfile'])
    # Load features for true paths
    featureims.data_from_file(
        filepath=params['intermedfolder'] + params['featureimsfile'],
        nodata=True
    )

    return (paths_true, paths_false, featureims)

def features_of_paths(ipl):

    params = ipl.get_params()
    thisparams = rdict(params['features_of_paths'])
    targetfile = params['intermedfolder'] + params['featuresfile']

    # Load the necessary images
    paths_true, paths_false, featureims = load_images(ipl)
    ipl.logging('\nInitial datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

    for d, k, v, kl in paths_true.data_iterator(yield_short_kl=True):

        if k == 'truepaths':

            ipl.logging('===============================\nWorking on group: {}', kl)

            # # TODO: Implement copy full logger
            # ipl[kl].set_logger(ipl.get_logger())

            # # Load the image data into memory
            # ipl[kl].populate()

            ipl[kl] = libip.features_of_paths(
                ipl,
                paths_true[kl]['truepaths'], paths_false[kl]['falsepaths'],
                featureims[kl], featureims[kl], kl
            )

            # Write the result to file
            ipl.write(filepath=targetfile, keys=[kl])
            # # Free memory
            ipl[kl] = None



def run_features_of_paths(yamlfile, logging=True):

    ipl = IPL(yaml=yamlfile)

    ipl.set_indent(1)

    params = rdict(data=ipl.get_params())
    if logging:
        ipl.startlogger(filename=params['resultfolder'] + 'features_of_paths.log', type='w', name='FeaturesOfPaths')
    else:
        ipl.startlogger()

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

    run_features_of_paths(yamlfile, logging=False)