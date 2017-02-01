
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

    all_params = yparams.get_params()

    # Zero'th layer:
    # --------------
    zeroth = rdict(all_params['random_forest'])
    if 'default' in zeroth:
        zeroth_defaults = zeroth.pop('default')
    else:
        zeroth_defaults = ipl()

    # pathlist = ipl()
    # pathlistfile = zeroth_defaults['targets', 'pathlist']
    # pathlistfile = all_params[pathlistfile[0]] + all_params[pathlistfile[1]]

    # yparams.logging('\nDatastructure of pathlistin:\n\n{}', pathlistin.datastructure2string())

    for exp_lbl, experiment in zeroth.iteritems():

        # First layer
        # -----------
        # An experiment is now selected and performed
        yparams.logging('\n\nPerforming experiment {}\n==============================', exp_lbl)

        final = zeroth_defaults.dcp()
        final.merge(experiment)

        exp_sources = final['sources']
        exp_params = final['params']
        exp_targets = final['targets']
        exp_source_kl = [exp_lbl]
        if len(exp_sources['train']) == 4:
            exp_source_kl = exp_sources['train'][3]
        exp_predict_kl = ['predict']
        if len(exp_sources['predict']) == 4:
            exp_predict_kl = exp_sources['predict'][3]
        if type(exp_source_kl) is str:
            exp_source_kl = [exp_source_kl]
        if type(exp_predict_kl) is str:
            exp_predict_kl = [exp_predict_kl]

        # Get the pathlist stored in features_of_paths
        pathlist_source = exp_sources.pop('pathlist')
        pathlistfile = all_params[pathlist_source[0]] \
                       + all_params[pathlist_source[1]]
        with open(pathlistfile, 'r') as f:
            pathlistin = pickle.load(f)

        if len(pathlist_source) > 2:
            if 'skeys' in pathlist_source[2]:
                pathlistin = pathlistin.subset(*pathlist_source[2]['skeys'])
                print 'I was here...'

        yparams.logging('pathlistin.datastructure: \n{}\n', pathlistin.datastructure2string(maxdepth=4))
        pathlistout = ipl()

        # Load training data
        if 'train' in exp_sources.keys():
            truesource = exp_sources['train']
            falsesource = exp_sources['train']
        else:
            truesource = exp_sources['traintrue']
            falsesource = exp_sources['trainfalse']
        truetrainfeats = load_data(
            all_params[truesource[0]] + all_params[truesource[1]], logger=yparams, **truesource[2]
        ).subset('truepaths', search=True)
        falsetrainfeats = load_data(
            all_params[falsesource[0]] + all_params[falsesource[1]], logger=yparams, **falsesource[2]
        ).subset('falsepaths', search=True)

        yparams.logging('\ntruetrainfeats.datastructure: \n{}\n', truetrainfeats.datastructure2string(maxdepth=4))
        yparams.logging('\nfalsetrainfeats.datastructure: \n{}\n', falsetrainfeats.datastructure2string(maxdepth=4))

        # Load prediction data
        predictsource = exp_sources['predict']
        predictfeats = load_data(
            all_params[predictsource[0]] + all_params[predictsource[1]],
            logger=yparams, **predictsource[2]
        )
        yparams.logging('\npredictfeats.datastructure: \n{}\n', predictfeats.datastructure2string(maxdepth=4))

        # Load the data into memory
        truetrainfeats.populate()
        falsetrainfeats.populate()
        predictfeats.populate()

        # Concatenate the different sources
        # 1. Of training data
        plo_true = ipl()
        truetrainfeats, plo_true['truepaths'] = libip.rf_combine_sources_new(
            truetrainfeats[exp_source_kl]['truepaths'].dcp(),
            pathlistin[exp_source_kl]['truepaths'].dcp()
        )
        plo_false = ipl()
        falsetrainfeats, plo_false['falsepaths'] = libip.rf_combine_sources_new(
            falsetrainfeats[exp_source_kl]['falsepaths'].dcp(),
            pathlistin[exp_source_kl]['falsepaths'].dcp()
        )
        pathlistout[exp_source_kl + ['train']] = plo_true + plo_false
        # 2. Of prediction data
        ipf_true = ipl()
        plo_true = ipl()
        ipf_true['truepaths'], plo_true['truepaths'] = libip.rf_combine_sources_new(
            predictfeats[exp_predict_kl]['truepaths'].dcp(),
            pathlistin[exp_predict_kl]['truepaths'].dcp()
        )
        ipf_false = ipl()
        plo_false = ipl()
        ipf_false['falsepaths'], plo_false['falsepaths'] = libip.rf_combine_sources_new(
            predictfeats[exp_predict_kl]['falsepaths'].dcp(),
            pathlistin[exp_predict_kl]['falsepaths'].dcp()
        )
        inpredictfeats = ipf_true + ipf_false
        pathlistout[exp_source_kl, 'predict'] = plo_true + plo_false

        # Note:
        #   Due to the feature input being a dictionary organized by the feature images where
        #   the feature values come from
        #
        #       [source]
        #           'truepaths'|'falsepaths'
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
        #   When accumulating the features for each of the four corresponding subsets, namely
        #   training and testing set with true and false paths each, i.e.
        #   ['0'|'1']['truefeats'|'falsefeats'],
        #   the the keylist is used, thus maintaining the correct order in every subset.
        #
        # And that is what is happening here:
        # #   1. Get the keylist of a full feature list, e.g. one of true paths
        # example_kl = None
        # for d2, k2, v2, kl2 in truetrainfeats.data_iterator():
        #     if k2 == 'truepaths':
        #         example_kl = kl2
        #         break
        # 2. Get the keylist order of the feature space
        feature_space_list = []
        for d2, k2, v2, kl2 in truetrainfeats.data_iterator():
            if type(v2) is not type(truetrainfeats):
                feature_space_list.append(kl2)

        intrain = ipl()
        intrain['true'] = libip.rf_make_feature_array_with_keylist(truetrainfeats, feature_space_list)
        yparams.logging("Computed feature array for train['true'] with shape {}", intrain['true'].shape)
        intrain['false'] = libip.rf_make_feature_array_with_keylist(falsetrainfeats, feature_space_list)
        yparams.logging("Computed feature array for train['false'] with shape {}", intrain['false'].shape)

        inpredictfeats['true'] = libip.rf_make_feature_array_with_keylist(inpredictfeats['truepaths'], feature_space_list)
        yparams.logging("Computed feature array for predict['true'] with shape {}", inpredictfeats['true'].shape)
        inpredictfeats['false'] = libip.rf_make_feature_array_with_keylist(inpredictfeats['falsepaths'], feature_space_list)
        yparams.logging("Computed feature array for predict['false'] with shape {}", inpredictfeats['false'].shape)

        # Classify
        result = ipl()
        result[exp_lbl] = libip.random_forest(
            intrain, inpredictfeats, debug=debug, balance=exp_params['balance_classes'],
            logger=yparams
        )

        # Evaluate
        new_eval = ipl()
        # print [x[0] for x in result[kl]]
        # print [x[1] for x in result[kl]]
        new_eval[exp_lbl] = libip.new_eval([x[0] for x in result[exp_lbl]], [x[1] for x in result[exp_lbl]])

        yparams.logging('+++ RESULTS +++')
        yparams.logging("[kl]")
        # for i in result[kl]:
        #     yparams.logging('{}', i)
        for key, value in new_eval[exp_lbl].iteritems():
            yparams.logging('{} = {}', key, value)

        pass

    sys.exit()


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
            yparams.logging("Computed feature array for train['true'] with shape {}", intrain['true'].shape)
            intrain['false'] = libip.rf_make_feature_array_with_keylist(falsetrainfeats['false'], feature_space_list)
            yparams.logging("Computed feature array for train['false'] with shape {}", intrain['false'].shape)

            inpredictfeats['true'] = libip.rf_make_feature_array_with_keylist(inpredictfeats['true'], feature_space_list)
            yparams.logging("Computed feature array for predict['true'] with shape {}", inpredictfeats['true'].shape)
            inpredictfeats['false'] = libip.rf_make_feature_array_with_keylist(inpredictfeats['false'], feature_space_list)
            yparams.logging("Computed feature array for predict['false'] with shape {}", inpredictfeats['false'].shape)

            # Classify
            result = ipl()
            result[kl] = libip.random_forest(
                intrain, inpredictfeats, debug=debug, balance=balance_classes, logger=yparams
            )

            # Evaluate
            new_eval = ipl()
            # print [x[0] for x in result[kl]]
            # print [x[1] for x in result[kl]]
            new_eval[kl] = libip.new_eval([x[0] for x in result[kl]], [x[1] for x in result[kl]])

            yparams.logging('+++ RESULTS +++')
            yparams.logging("[kl]")
            # for i in result[kl]:
            #     yparams.logging('{}', i)
            for key, value in new_eval[kl].iteritems():
                yparams.logging('{} = {}', key, value)

            # # # Write the result to file
            # # ipl.write(filepath=targetfile, keys=[kl])
            # # Free memory
            # data[kl] = None

    # def val(x):
    #     return x
    # pathlistout.dss(function=val)


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

    # yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters_ref.yml'
    yamlfile = '/mnt/localdata01/jhennies/neuraldata/results/cremi_2016/170127_only_on_beta_5_train0_predict1_full/parameters.yml'

    run_random_forest(yamlfile, logging=False, make_only_feature_array=False, debug=False, write=False)
