
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
    These datasets are loaded:
    features
    :param ipl:
    :return:
    """

    params = ipl.get_params()

    # Paths within labels (true paths)
    ipl.data_from_file(
        filepath=params['intermedfolder'] + params['featuresfile'],nodata=True
    )


def random_forest(ipl):

    params = ipl.get_params()
    thisparams = rdict(params['random_forest'])
    targetfile = params['resultfolder'] + params['resultsfile']

    # Load the necessary images
    load_images(ipl)
    ipl.logging('\nInitial datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

    for d, k, v, kl in ipl.data_iterator(yield_short_kl=True):

        if k == 'true':

            ipl.logging('===============================\nWorking on group: {}', kl)

            # TODO: Implement copy full logger
            ipl[kl].set_logger(ipl.get_logger())

            # # Load the image data into memory
            # ipl[kl].populate()

            a = IPL()
            a['true'] = libip.rf_make_feature_array(ipl[kl]['true'])

            # # Write the result to file
            # ipl.write(filepath=targetfile, keys=[kl])
            # # Free memory
            # ipl[kl] = None



def run_random_forest(yamlfile):

    ipl = IPL(yaml=yamlfile)

    ipl.set_indent(1)

    params = rdict(data=ipl.get_params())
    ipl.startlogger(filename=params['resultfolder'] + 'random_forest.log', type='w', name='RandomForest')

    try:

        # Copy the script file and the parameters to the scriptsfolder
        copy(inspect.stack()[0][1], params['scriptsfolder'])
        copy(yamlfile, params['scriptsfolder'] + 'random_forest.parameters.yml')

        # ipl.logging('\nInitial datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

        random_forest(ipl)

        # ipl.logging('\nFinal datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

        # features.write(filepath=params['intermedfolder'] + params['featuresfile'])

        ipl.logging('')
        ipl.stoplogger()

    except:
        ipl.errout('Unexpected error')


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'

    run_random_forest(yamlfile)