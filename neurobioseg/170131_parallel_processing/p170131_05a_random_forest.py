
import os
import inspect
# from hdf5_image_processing import Hdf5ImageProcessing as IP, Hdf5ImageProcessingLib as ipl
from hdf5_slim_processing import Hdf5Processing as hp
# from hdf5_processing import RecursiveDict as rdict
from hdf5_slim_processing import RecursiveDict as Rdict
from shutil import copy, copyfile
import numpy as np
import matplotlib.pyplot as plt
# import processing_libip as libip
import slim_processing_libhp as libhp
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

    data = hp()

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
    zeroth = Rdict(all_params['random_forest'])
    if 'default' in zeroth:
        zeroth_defaults = zeroth.pop('default')
    else:
        zeroth_defaults = hp()

    # pathlist = ipl()
    featlistfile = zeroth_defaults['targets', 'featlist']
    featlistfile = all_params[featlistfile[0]] + all_params[featlistfile[1]]

    # yparams.logging('\nDatastructure of pathlistin:\n\n{}', pathlistin.datastructure2string())

    feature_space_lists = dict()

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

        # Loading of the training pathlist(s)
        # --------------------------
        # Get the pathlist stored in features_of_paths
        pathlist_source = exp_sources.pop('train_pl')

        # Check for list or single file
        if type(pathlist_source) is Rdict:
            pathlistin_train = Rdict()
            for key, val in pathlist_source.iteritems():
                pathlistfile = all_params[val[0]] + all_params[val[1]]
                with open(pathlistfile, 'r') as f:
                    pathlistin_train[key] = Rdict(pickle.load(f))
                if 'skeys' in val[2]:
                    pathlistin_train[key] = pathlistin_train[key].subset(*val[2]['skeys'])
        else:
            pathlistfile = all_params[pathlist_source[0]] \
                           + all_params[pathlist_source[1]]
            with open(pathlistfile, 'r') as f:
                pathlistin_train = Rdict(pickle.load(f))
            if 'skeys' in pathlist_source[2]:
                pathlistin_train = pathlistin_train.subset(*pathlist_source[2]['skeys'])
        yparams.logging('pathlistin_train.datastructure: \n{}\n', pathlistin_train.datastructure2string(maxdepth=4))
        pathlistout = hp()

        # Loading of the prediction pathlist
        pathlist_source = exp_sources.pop('predict_pl')

        pathlistfile = all_params[pathlist_source[0]] \
                       + all_params[pathlist_source[1]]
        with open(pathlistfile, 'r') as f:
            pathlistin_predict = Rdict(pickle.load(f))
        if 'skeys' in pathlist_source[2]:
            pathlistin_predict = pathlistin_predict.subset(*pathlist_source[2]['skeys'])
        yparams.logging('pathlistin_predict.datastructure: \n{}\n', pathlistin_predict.datastructure2string(maxdepth=4))

        # Load training data
        # ------------------

        if 'train' in exp_sources.keys():
            truesource = exp_sources['train']
            falsesource = exp_sources['train']
        else:
            truesource = exp_sources['traintrue']
            falsesource = exp_sources['trainfalse']

        # Check for list or single file
        if type(truesource) is Rdict:
            truetrainfeats = hp()
            for key, val in truesource.iteritems():
                truetrainfeats[key] = load_data(
                    all_params[val[0]] + all_params[val[1]], logger=yparams, **val[2]
                ).subset('truepaths', search=True)
        else:
            truetrainfeats = load_data(
                all_params[truesource[0]] + all_params[truesource[1]], logger=yparams, **truesource[2]
            ).subset('truepaths', search=True)
        if type(falsesource) is Rdict:
            falsetrainfeats = hp()
            for key, val in falsesource.iteritems():
                falsetrainfeats[key] = load_data(
                    all_params[val[0]] + all_params[val[1]], logger=yparams, **val[2]
                ).subset('falsepaths', search=True)
        else:
            falsetrainfeats = load_data(
                all_params[falsesource[0]] + all_params[falsesource[1]], logger=yparams, **falsesource[2]
            ).subset('falsepaths', search=True)

        # ------------------

        yparams.logging('\ntruetrainfeats.datastructure: \n{}\n', truetrainfeats.datastructure2string(maxdepth=4))
        yparams.logging('\nfalsetrainfeats.datastructure: \n{}\n', falsetrainfeats.datastructure2string(maxdepth=4))

        # Load prediction data
        predictsource = exp_sources['predict']
        predictfeats = load_data(
            all_params[predictsource[0]] + all_params[predictsource[1]],
            logger=yparams, **predictsource[2]
        )
        yparams.logging('\npredictfeats.datastructure: \n{}\n', predictfeats.datastructure2string(maxdepth=4))

        # # Load the data into memory
        # truetrainfeats.populate()
        # falsetrainfeats.populate()
        # predictfeats.populate()

        # Concatenate the different sources
        # 1. Of training data
        plo_true_train = hp()
        plo_false_train = hp()
        # truetrainfeats, plo_true['truepaths'] = libip.rf_combine_sources_new(
        #     truetrainfeats[exp_source_kl]['truepaths'].dcp(),
        #     pathlistin[exp_source_kl]['truepaths'].dcp()
        # )
        truetrainfeats, plo_true_train['train', 'truepaths'] = libhp.rf_combine_sources(
            truetrainfeats,
            pathlistin_train.subset('truepaths', search=True)
        )
        falsetrainfeats, plo_false_train['train', 'falsepaths'] = libhp.rf_combine_sources(
            falsetrainfeats,
            pathlistin_train.subset('falsepaths', search=True)
        )
        pathlistout[exp_source_kl] = plo_true_train + plo_false_train
        # 2. Of prediction data
        ipf_true = hp()
        plo_true_predict = hp()
        ipf_true['truepaths'], plo_true_predict['predict', 'truepaths'] = libhp.rf_combine_sources(
            predictfeats.subset('truepaths', search=True),
            pathlistin_predict.subset('truepaths', search=True)
        )
        ipf_false = hp()
        plo_false_predict = hp()
        ipf_false['falsepaths'], plo_false_predict['predict', 'falsepaths'] = libhp.rf_combine_sources(
            predictfeats.subset('falsepaths', search=True),
            pathlistin_predict.subset('falsepaths', search=True)
        )
        inpredictfeats = ipf_true + ipf_false
        pathlistout[exp_source_kl, 'predict'] = plo_true_predict + plo_false_predict

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
        # TODO: Write this to file
        feature_space_list = []
        for d2, k2, v2, kl2 in truetrainfeats.data_iterator():
            if type(v2) is not type(truetrainfeats):
                feature_space_list.append(kl2)

        feature_space_lists[exp_lbl] = feature_space_list

        intrain = hp()
        intrain['true'] = libhp.rf_make_feature_array_with_keylist(truetrainfeats, feature_space_list)
        yparams.logging("Computed feature array for train['true'] with shape {}", intrain['true'].shape)
        intrain['false'] = libhp.rf_make_feature_array_with_keylist(falsetrainfeats, feature_space_list)
        yparams.logging("Computed feature array for train['false'] with shape {}", intrain['false'].shape)

        inpredict = hp()
        inpredict['true'] = libhp.rf_make_feature_array_with_keylist(inpredictfeats['truepaths'], feature_space_list)
        yparams.logging("Computed feature array for predict['true'] with shape {}", inpredict['true'].shape)
        inpredict['false'] = libhp.rf_make_feature_array_with_keylist(inpredictfeats['falsepaths'], feature_space_list)
        yparams.logging("Computed feature array for predict['false'] with shape {}", inpredict['false'].shape)

        # Classify
        result = hp()
        result[exp_lbl] = libhp.random_forest(
            intrain, inpredict, debug=debug, balance=exp_params['balance_classes'],
            logger=yparams
        )

        # Evaluate
        new_eval = hp()
        # print [x[0] for x in result[kl]]
        # print [x[1] for x in result[kl]]
        new_eval[exp_lbl] = libhp.new_eval([x[0] for x in result[exp_lbl]], [x[1] for x in result[exp_lbl]])

        yparams.logging('+++ RESULTS +++')
        yparams.logging("[kl]")
        # for i in result[kl]:
        #     yparams.logging('{}', i)
        for key, value in new_eval[exp_lbl].iteritems():
            yparams.logging('{} = {}', key, value)

    with open(featlistfile, 'wb') as f:
        pickle.dump(feature_space_lists, f)


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
    # yamlfile = '/mnt/localdata01/jhennies/neuraldata/results/cremi_2016/170127_only_on_beta_5_train0_predict1_full/parameters.yml'
    # yamlfile = '/mnt/localdata01/jhennies/neuraldata/results/cremi_2016/170216_crossvalidation_spl_abc/parameters.yml'

    run_random_forest(yamlfile, logging=False, make_only_feature_array=False, debug=True, write=False)
