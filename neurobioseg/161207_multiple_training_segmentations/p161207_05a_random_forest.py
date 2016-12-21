
import os
import inspect
from hdf5_image_processing import Hdf5ImageProcessing as IP, Hdf5ImageProcessingLib as ipl
from hdf5_processing import RecursiveDict as rdict
from shutil import copy, copyfile
import numpy as np
import matplotlib.pyplot as plt
import processing_libip as libip
import sys
from yaml_parameters import YamlParams
import pickle


__author__ = 'jhennies'


# def load_images(ipl):
#     """
#     These datasets are loaded:
#     features
#     :param ipl:
#     :return:
#     """
#
#     params = ipl.get_params()
#
#     # Paths within labels (true paths)
#     ipl.logging('Loading features ...')
#     ipl.logging('   File path = {}', params['intermedfolder'] + params['featuresfile'])
#     ipl.data_from_file(
#         filepath=params['intermedfolder'] + params['featuresfile'],
#         nodata=True
#     )


def load_data(filepath, skeys=None, recursive_search=False, logger=None):

    if logger is not None:
        logger.logging('Loading data from \n{}', filepath)
    else:
        print 'Loading data from \n{}'.format(filepath)

    data = ipl()

    data.data_from_file(
        filepath=filepath,
        skeys=skeys,
        recursive_search=recursive_search,
        nodata=True
    )

    return data


def random_forest(yparams, debug=False):

    params = yparams.get_params()
    thisparams = rdict(params['random_forest'])

    # Get the pathlist stored in features_of_paths
    pathlistfile = params[thisparams['pathlistin'][0]] \
                 + params[thisparams['pathlistin'][1]]
    with open(pathlistfile, 'r') as f:
        pathlistin = pickle.load(f)
    pathlistout = ipl()

    yparams.logging('\nDatastructure of pathlistin:\n\n{}', pathlistin.datastructure2string())

    # for i in xrange(0, len(thisparams['sources'])):
    for d, k, v, kl in thisparams['sources'].data_iterator(yield_short_kl=True):

        if k == 'predict':

            yparams.logging('===============================\nWorking on group: {}', kl)

            # Get parameters (currently only 'balance_classes') and set defaults
            balance_classes = False
            if 'balance_classes' in thisparams['sources'][kl].keys():
                balance_classes = thisparams['sources'][kl]['balance_classes']

            # Load training data
            if 'train' in thisparams['sources'][kl].keys():
                truesource = thisparams['sources'][kl]['train']
                falsesource = thisparams['sources'][kl]['train']
            else:
                truesource = thisparams['sources'][kl]['traintrue']
                falsesource = thisparams['sources'][kl]['trainfalse']
            truetrainfeats = load_data(
                params[truesource[0]] + params[truesource[1]], logger=yparams, **truesource[2]
            ).subset('true', search=True)
            falsetrainfeats = load_data(
                params[falsesource[0]] + params[falsesource[1]], logger=yparams, **falsesource[2]
            ).subset('false', search=True)
            # # The plus operator is overloaded to perform a merging operation on RecursiveDicts
            # trainfeats = truetrainfeats + falsetrainfeats
            # yparams.logging(
            #     '\nDatastructure of truetrainfeats\n\n{}',
            #     truetrainfeats.datastructure2string(maxdepth=3)
            # )
            # yparams.logging(
            #     '\nDatastructure of falsetrainfeats\n\n{}',
            #     falsetrainfeats.datastructure2string(maxdepth=3)
            # )

            # Load prediction data
            predictsource = thisparams['sources'][kl]['predict']
            predictfeats = load_data(
                params[predictsource[0]] + params[predictsource[1]],
                logger=yparams, **predictsource[2]
            )

            # Note:
            #   Due to the feature input being a dictionary organized by the feature images where
            #   the feature values come from
            #
            #       [source]
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
            #   1. Get the keylist of a full feature list, e.g. one of true paths
            example_kl = None
            for d2, k2, v2, kl2 in truetrainfeats.data_iterator():
                if k2 == 'true':
                    example_kl = kl2
                    break
            #   2. Get the keylist order of the feature space
            feature_space_list = []
            for d2, k2, v2, kl2 in truetrainfeats[example_kl].data_iterator():
                if type(v2) is not type(truetrainfeats[example_kl]):
                    feature_space_list.append(kl2)
            # yparams.logging('feature_space_list[i] = {}', feature_space_list, listed=True)

            # Load the data into memory
            truetrainfeats.populate()
            falsetrainfeats.populate()
            predictfeats.populate()

            truetrainfeats, plo_true = libip.rf_combine_sources(
                truetrainfeats, search_for='true', pathlist=pathlistin
            )
            falsetrainfeats, plo_false = libip.rf_combine_sources(
                falsetrainfeats, search_for='false', pathlist=pathlistin
            )
            pathlistout[kl + ['train']] = plo_true + plo_false

            ipf_true, plo_true = libip.rf_combine_sources(
                predictfeats, search_for='true', pathlist=pathlistin
            )
            ipf_false, plo_false = libip.rf_combine_sources(
                predictfeats, search_for='false', pathlist=pathlistin
            )
            inpredictfeats = ipf_true + ipf_false
            pathlistout[kl + ['predict']] = plo_true + plo_false

            # yparams.logging(
            #     '\nDatastructure of truetrainfeats\n\n{}',
            #     truetrainfeats.datastructure2string(maxdepth=3)
            # )
            # yparams.logging(
            #     '\nDatastructure of falsetrainfeats\n\n{}',
            #     falsetrainfeats.datastructure2string(maxdepth=3)
            # )

            intrain = ipl()
            intrain['true'] = libip.rf_make_feature_array_with_keylist(truetrainfeats['true'], feature_space_list)
            yparams.logging("Computed feature array for ['true'] with shape {}", intrain['true'].shape)
            intrain['false'] = libip.rf_make_feature_array_with_keylist(falsetrainfeats['false'], feature_space_list)
            yparams.logging("Computed feature array for ['false'] with shape {}", intrain['false'].shape)

            inpredictfeats['true'] = libip.rf_make_feature_array_with_keylist(inpredictfeats['true'], feature_space_list)
            yparams.logging("Computed feature array for ['true'] with shape {}", inpredictfeats['true'].shape)
            inpredictfeats['false'] = libip.rf_make_feature_array_with_keylist(inpredictfeats['false'], feature_space_list)
            yparams.logging("Computed feature array for ['false'] with shape {}", inpredictfeats['false'].shape)

            # Classify
            result = ipl()
            result[kl] = libip.random_forest(intrain, inpredictfeats, debug=debug)

            # Evaluate
            new_eval = ipl()
            new_eval[kl] = libip.new_eval([x[0] for x in result[kl]], [x[1] for x in result[kl]])

            yparams.logging('+++ RESULTS +++')
            yparams.logging("[kl]")
            for i in result[kl]:
                yparams.logging('{}', i)
            for key, value in new_eval[kl].iteritems():
                yparams.logging('{} = {}', key, value)

            # # # Write the result to file
            # # ipl.write(filepath=targetfile, keys=[kl])
            # # Free memory
            # data[kl] = None

    def val(x):
        return x
    pathlistout.dss(function=val)

    # params = ipl.get_params()
    # thisparams = rdict(params['random_forest'])
    # targetfile = params['resultfolder'] + params['resultsfile']
    #
    # # Load the necessary images
    # load_images(data)
    # ipl.logging('\nInitial datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))
    #
    # result = ipl()
    # new_eval = rdict()
    # evaluation = rdict()
    #
    # for d, k, v, kl in data.data_iterator(yield_short_kl=True):
    #
    #     if k == '0':
    #
    #         data.logging('===============================\nWorking on group: {}', kl)
    #
    #         # TODO: Implement copy full logger
    #         data[kl].set_logger(data.get_logger())
    #
    #         # Note:
    #         #   Due to the feature input being a dictionary organized by the feature images where
    #         #   the feature values come from
    #         #
    #         #   [kl]
    #         #       '0'|'1'
    #         #           'true'|'false'
    #         #               [featureims]
    #         #                   'Sum':      [s1, ..., sN]
    #         #                   'Variance': [v1, ..., vN]
    #         #                   ...
    #         #               [Pathlength]:   [l1, ..., lN]
    #         #
    #         #   the exact order in which items are iterated over by data_iterator() is not known.
    #         #
    #         # Solution:
    #         #   Iterate over it once and store the keylist in an array (which conserves the order)
    #         #   When accumulating the featrues for each of the four corresponding subsets, namely
    #         #   training and testing set with true and false paths each, i.e.
    #         #   ['0'|'1']['true'|'false'],
    #         #   the the keylist is used, thus maintaining the correct order in every subset.
    #         #
    #         # And that is what is happening here:
    #         keylist = []
    #         for d2, k2, v2, kl2 in data[kl]['0', 'true'].data_iterator():
    #             if type(v2) is not type(data[kl]['0', 'true']):
    #                 keylist.append(kl2)
    #
    #         # Load the image data into memory
    #         data[kl].populate()
    #
    #         # ipl[kl]['0', 'true'] = libip.rf_make_feature_array(ipl[kl]['0', 'true'])
    #         # ipl.logging("Computed feature array for ['0', 'true'] with shape {}", ipl[kl]['0', 'true'].shape)
    #         # ipl[kl]['0', 'false'] = libip.rf_make_feature_array(ipl[kl]['0', 'false'])
    #         # ipl.logging("Computed feature array for ['0', 'false'] with shape {}", ipl[kl]['0', 'false'].shape)
    #         # ipl[kl]['1', 'true'] = libip.rf_make_feature_array(ipl[kl]['1', 'true'])
    #         # ipl.logging("Computed feature array for ['1', 'true'] with shape {}", ipl[kl]['1', 'true'].shape)
    #         # ipl[kl]['1', 'false'] = libip.rf_make_feature_array(ipl[kl]['1', 'false'])
    #         # ipl.logging("Computed feature array for ['1', 'false'] with shape {}", ipl[kl]['1', 'false'].shape)
    #
    #         data[kl]['0', 'true'] = libip.rf_make_feature_array_with_keylist(data[kl]['0', 'true'], keylist)
    #         data.logging("Computed feature array for ['0', 'true'] with shape {}", data[kl]['0', 'true'].shape)
    #         data[kl]['0', 'false'] = libip.rf_make_feature_array_with_keylist(data[kl]['0', 'false'], keylist)
    #         data.logging("Computed feature array for ['0', 'false'] with shape {}", data[kl]['0', 'false'].shape)
    #         data[kl]['1', 'true'] = libip.rf_make_feature_array_with_keylist(data[kl]['1', 'true'], keylist)
    #         data.logging("Computed feature array for ['1', 'true'] with shape {}", data[kl]['1', 'true'].shape)
    #         data[kl]['1', 'false'] = libip.rf_make_feature_array_with_keylist(data[kl]['1', 'false'], keylist)
    #         data.logging("Computed feature array for ['1', 'false'] with shape {}", data[kl]['1', 'false'].shape)
    #
    #         # print '...'
    #         # print ipl[kl]['0']
    #
    #         result[kl + ['0']] = libip.random_forest(data[kl]['0'], ipl[kl]['1'], debug=debug)
    #         result[kl + ['1']] = libip.random_forest(data[kl]['1'], ipl[kl]['0'], debug=debug)
    #
    #         new_eval[kl + ['0']] = libip.new_eval([x[0] for x in result[kl]['0']], [x[1] for x in result[kl]['0']])
    #         new_eval[kl + ['1']] = libip.new_eval([x[0] for x in result[kl]['1']], [x[1] for x in result[kl]['1']])
    #
    #         evaluation[kl + ['0']] = libip.evaluation(result[kl]['0'])
    #         evaluation[kl + ['1']] = libip.evaluation(result[kl]['1'])
    #
    #         data.logging('+++ RESULTS +++')
    #         data.logging("[kl]['0']")
    #         for i in result[kl]['0']:
    #             data.logging('{}', i)
    #         # for key, value in evaluation[kl]['0'].iteritems():
    #         #     ipl.logging('{} = {}', key, value)
    #         for key, value in new_eval[kl]['0'].iteritems():
    #             data.logging('{} = {}', key, value)
    #
    #         data.logging('+++')
    #         data.logging("[kl]['1']")
    #         for i in result[kl]['1']:
    #             data.logging('{}', i)
    #         # for key, value in evaluation[kl]['1'].iteritems():
    #         #     ipl.logging('{} = {}', key, value)
    #         for key, value in new_eval[kl]['1'].iteritems():
    #             data.logging('{} = {}', key, value)
    #
    #         # # Write the result to file
    #         # ipl.write(filepath=targetfile, keys=[kl])
    #         # Free memory
    #             data[kl] = None
    #
    # return ipl(data=result), ipl(data=evaluation)


# def run_random_forest(yamlfile, logging=True, make_only_feature_array=False, debug=False, write=True):
#
#     data = ipl(yaml=yamlfile)
#
#     data.set_indent(1)
#
#     params = rdict(data=ipl.get_params())
#     if logging:
#         data.startlogger(filename=params['resultfolder'] + 'random_forest.log', type='w', name='RandomForest')
#     else:
#         data.startlogger()
#
#     try:
#
#         # # Copy the script file and the parameters to the scriptsfolder
#         # copy(inspect.stack()[0][1], params['scriptsfolder'])
#         # copy(yamlfile, params['scriptsfolder'] + 'random_forest.parameters.yml')
#
#         # ipl.logging('\nInitial datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))
#
#         if make_only_feature_array:
#             make_feature_arrays(ipl)
#         else:
#             result = ipl()
#             result['result'], result['evaluation'] = random_forest(data, debug=debug)
#
#             # ipl.logging('\nFinal datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))
#
#             if write:
#                 result.write(filepath=params['resultfolder'] + params['resultsfile'])
#
#         data.logging('')
#         data.stoplogger()
#
#     except:
#         data.errout('Unexpected error')


def run_random_forest(
        yamlfile, logging=True, make_only_feature_array=False, debug=False, write=True
):

    yparams = YamlParams(filename=yamlfile)
    params = yparams.get_params()

    # Logger stuff
    yparams.set_indent(1)
    yparams.startlogger(
        filename=params['resultfolder'] + 'random_forest.log',
        type='w', name='RandomForest'
    )

    try:

        random_forest(yparams, debug)

        yparams.logging('')
        yparams.stoplogger()

    except:
        raise
        yparams.errout('Unexpected error')


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters_ref.yml'

    run_random_forest(yamlfile, logging=False, make_only_feature_array=False, debug=True, write=False)