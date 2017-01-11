
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


# This function is for debugging purposes
def make_feature_arrays(ipl):

    params = ipl.get_params()
    thisparams = rdict(params['random_forest'])
    targetfile = params['resultfolder'] + params['resultsfile']

    # Load the necessary images
    load_images(ipl)
    ipl.logging('\nInitial datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

    result = IPL()
    evaluation = rdict()

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

            ipl.write(filepath=params['intermedfolder'] + 'feature_arrays.h5', keys=[kl])


def random_forest(ipl, debug=False):

    params = ipl.get_params()
    thisparams = rdict(params['random_forest'])
    targetfile = params['resultfolder'] + params['resultsfile']

    # Load the necessary images
    load_images(ipl)
    ipl.logging('\nInitial datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

    result = IPL()
    new_eval = rdict()
    evaluation = rdict()

    for d, k, v, kl in ipl.data_iterator(yield_short_kl=True):

        if k == '0':

            ipl.logging('===============================\nWorking on group: {}', kl)

            # TODO: Implement copy full logger
            ipl[kl].set_logger(ipl.get_logger())

            # Note:
            #   Due to the feature input being a dictionary organized by the feature images where
            #   the feature values come from
            #
            #   [kl]
            #       '0'|'1'
            #           'true'|'false'
            #               [featureims]
            #                   'Sum':      [s1, ..., sN]
            #                   'Variance': [v1, ..., vN]
            #                   ...
            #               [Pathlength]:   [l1, ..., lN]
            #
            #   the exact order in which items are iterated over by data_iterator() is not known.
            #
            # Solution:
            #   Iterate over it once and store the keylist in an array (which conserves the order)
            #   When accumulating the featrues for each of the four corresponding subsets, namely
            #   training and testing set with true and false paths each, i.e.
            #   ['0'|'1']['true'|'false'],
            #   the the keylist is used, thus maintaining the correct order in every subset.
            #
            # And that is what is happening here:
            keylist = []
            for d2, k2, v2, kl2 in ipl[kl]['0', 'true'].data_iterator():
                if type(v2) is not type(ipl[kl]['0', 'true']):
                    keylist.append(kl2)

            # Load the image data into memory
            ipl[kl].populate()

            # ipl[kl]['0', 'true'] = libip.rf_make_feature_array(ipl[kl]['0', 'true'])
            # ipl.logging("Computed feature array for ['0', 'true'] with shape {}", ipl[kl]['0', 'true'].shape)
            # ipl[kl]['0', 'false'] = libip.rf_make_feature_array(ipl[kl]['0', 'false'])
            # ipl.logging("Computed feature array for ['0', 'false'] with shape {}", ipl[kl]['0', 'false'].shape)
            # ipl[kl]['1', 'true'] = libip.rf_make_feature_array(ipl[kl]['1', 'true'])
            # ipl.logging("Computed feature array for ['1', 'true'] with shape {}", ipl[kl]['1', 'true'].shape)
            # ipl[kl]['1', 'false'] = libip.rf_make_feature_array(ipl[kl]['1', 'false'])
            # ipl.logging("Computed feature array for ['1', 'false'] with shape {}", ipl[kl]['1', 'false'].shape)

            ipl[kl]['0', 'true'] = libip.rf_make_feature_array_with_keylist(ipl[kl]['0', 'true'], keylist)
            ipl.logging("Computed feature array for ['0', 'true'] with shape {}", ipl[kl]['0', 'true'].shape)
            ipl[kl]['0', 'false'] = libip.rf_make_feature_array_with_keylist(ipl[kl]['0', 'false'], keylist)
            ipl.logging("Computed feature array for ['0', 'false'] with shape {}", ipl[kl]['0', 'false'].shape)
            ipl[kl]['1', 'true'] = libip.rf_make_feature_array_with_keylist(ipl[kl]['1', 'true'], keylist)
            ipl.logging("Computed feature array for ['1', 'true'] with shape {}", ipl[kl]['1', 'true'].shape)
            ipl[kl]['1', 'false'] = libip.rf_make_feature_array_with_keylist(ipl[kl]['1', 'false'], keylist)
            ipl.logging("Computed feature array for ['1', 'false'] with shape {}", ipl[kl]['1', 'false'].shape)

            # print '...'
            # print ipl[kl]['0']

            result[kl + ['0']] = libip.random_forest(ipl[kl]['0'], ipl[kl]['1'], debug=debug)
            result[kl + ['1']] = libip.random_forest(ipl[kl]['1'], ipl[kl]['0'], debug=debug)

            new_eval[kl + ['0']] = libip.new_eval([x[0] for x in result[kl]['0']], [x[1] for x in result[kl]['0']])
            new_eval[kl + ['1']] = libip.new_eval([x[0] for x in result[kl]['1']], [x[1] for x in result[kl]['1']])

            evaluation[kl + ['0']] = libip.evaluation(result[kl]['0'])
            evaluation[kl + ['1']] = libip.evaluation(result[kl]['1'])

            ipl.logging('+++ RESULTS +++')
            ipl.logging("[kl]['0']")
            for i in result[kl]['0']:
                ipl.logging('{}', i)
            # for key, value in evaluation[kl]['0'].iteritems():
            #     ipl.logging('{} = {}', key, value)
            for key, value in new_eval[kl]['0'].iteritems():
                ipl.logging('{} = {}', key, value)

            ipl.logging('+++')
            ipl.logging("[kl]['1']")
            for i in result[kl]['1']:
                ipl.logging('{}', i)
            # for key, value in evaluation[kl]['1'].iteritems():
            #     ipl.logging('{} = {}', key, value)
            for key, value in new_eval[kl]['1'].iteritems():
                ipl.logging('{} = {}', key, value)

            # # Write the result to file
            # ipl.write(filepath=targetfile, keys=[kl])
            # Free memory
            ipl[kl] = None

    return IPL(data=result), IPL(data=evaluation)


def run_random_forest(yamlfile, logging=True, make_only_feature_array=False, debug=False, write=True):

    ipl = IPL(yaml=yamlfile)

    ipl.set_indent(1)

    params = rdict(data=ipl.get_params())
    if logging:
        ipl.startlogger(filename=params['resultfolder'] + 'random_forest.log', type='w', name='RandomForest')
    else:
        ipl.startlogger()

    try:

        # # Copy the script file and the parameters to the scriptsfolder
        # copy(inspect.stack()[0][1], params['scriptsfolder'])
        # copy(yamlfile, params['scriptsfolder'] + 'random_forest.parameters.yml')

        # ipl.logging('\nInitial datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

        if make_only_feature_array:
            make_feature_arrays(ipl)
        else:
            result = IPL()
            result['result'], result['evaluation'] = random_forest(ipl, debug=debug)

            # ipl.logging('\nFinal datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

            if write:
                result.write(filepath=params['resultfolder'] + params['resultsfile'])

        ipl.logging('')
        ipl.stoplogger()

    except:
        ipl.errout('Unexpected error')


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters_ref.yml'

    run_random_forest(yamlfile, logging=False, make_only_feature_array=False, debug=True, write=False)