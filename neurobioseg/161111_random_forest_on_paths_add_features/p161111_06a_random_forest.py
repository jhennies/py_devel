
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
    ipl.logging('Loading features ...')
    ipl.logging('   File path = {}', params['intermedfolder'] + params['featuresfile'])
    ipl.data_from_file(
        filepath=params['intermedfolder'] + params['featuresfile'],
        nodata=True
    )


def random_forest(ipl):

    params = ipl.get_params()
    thisparams = rdict(params['random_forest'])
    targetfile = params['resultfolder'] + params['resultsfile']

    # Load the necessary images
    load_images(ipl)
    ipl.logging('\nInitial datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

    result = IPL()

    for d, k, v, kl in ipl.data_iterator(yield_short_kl=True):

        if k == '0':

            ipl.logging('===============================\nWorking on group: {}', kl)

            # TODO: Implement copy full logger
            ipl[kl].set_logger(ipl.get_logger())

            # Load the image data into memory
            ipl[kl].populate()

            # def shp(x):
            #     return x.shape

            # print ipl[kl]['0', 'true']
            # print ipl[kl].dss(function=shp)

            ipl[kl]['0', 'true'] = libip.rf_make_feature_array(ipl[kl]['0', 'true'])
            ipl.logging("Computed feature array for ['0', 'true'] with shape {}", ipl[kl]['0', 'true'].shape)
            ipl[kl]['0', 'false'] = libip.rf_make_feature_array(ipl[kl]['0', 'false'])
            ipl.logging("Computed feature array for ['0', 'false'] with shape {}", ipl[kl]['0', 'false'].shape)
            ipl[kl]['1', 'true'] = libip.rf_make_feature_array(ipl[kl]['1', 'true'])
            ipl.logging("Computed feature array for ['1', 'true'] with shape {}", ipl[kl]['1', 'true'].shape)
            ipl[kl]['1', 'false'] = libip.rf_make_feature_array(ipl[kl]['1', 'false'])
            ipl.logging("Computed feature array for ['1', 'false'] with shape {}", ipl[kl]['1', 'false'].shape)

            # print '...'
            # print ipl[kl]['0']

            result[kl + ['0']] = libip.random_forest(ipl[kl]['0'], ipl[kl]['1'])
            result[kl + ['1']] = libip.random_forest(ipl[kl]['1'], ipl[kl]['0'])

            ipl.logging("[kl]['0']")
            for i in result[kl]['0']:
                ipl.logging('{}', i)
            ipl.logging("[kl]['1']")
            for i in result[kl]['1']:
                ipl.logging('{}', i)

            # # Write the result to file
            # ipl.write(filepath=targetfile, keys=[kl])
            # Free memory
            ipl[kl] = None

    return result


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

        result = random_forest(ipl)

        # ipl.logging('\nFinal datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

        # features.write(filepath=params['intermedfolder'] + params['featuresfile'])

        ipl.logging('')
        ipl.stoplogger()

    except:
        ipl.errout('Unexpected error')


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'

    run_random_forest(yamlfile)