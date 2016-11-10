
from hdf5_image_processing import Hdf5ImageProcessingLib as IPL
import vigra
from vigra.learning import RandomForest as RF
from vigra.learning import RandomForestOld as RFO
# Sklearn
from sklearn.ensemble import RandomForestClassifier as Skrf
import os
import sys
from shutil import copy
import inspect
import random

import numpy as np


__author__ = 'jhennies'


def eliminate_invalid_entries(data):

    data[np.isnan(data)] = 0
    data[np.isinf(data)] = 0

    data = data.astype(np.float32)

    return data

def concatenate_feature_arrays(features):
    """
    Concatenate feature array to gain a list of more entries
    :param features:
    :return:
    """
    features = np.concatenate(features.values(), axis=0)
    return features


def combine_feature_arrays(features):
    """
    Combine arrays to gain entries with more features
    :param features:
    :return:
    """
    for k, v in features.iteritems():
        features[k] = np.concatenate(v.values(), axis=1)

    return features


def features_to_array(features):
    """
    Make an array from the datastructure
    :param features:
    :return:
    """
    for d, k, v, kl in features.data_iterator(maxdepth=1):
        if d == 1:
            features[kl] = np.array(map(list, zip(*v.values())))

    return features


def make_feature_array(features):

    features_to_array(features)
    combine_feature_arrays(features)
    # From here on we need to use the return value since features changes type
    features = concatenate_feature_arrays(features)
    return features


def make_forest_input(features):

    lentrue = features['true'].shape[0]
    lenfalse = features['false'].shape[0]

    classes = np.concatenate((np.ones((lentrue,)), np.zeros((lenfalse,))))

    data = np.concatenate((features['true'], features['false']), axis=0)

    data = eliminate_invalid_entries(data)

    return [data, classes]


def random_forest(trainfeatures, testfeatures):

    # Done: For a first test, use half for training and half for testing
    # Done: Compute the class true features

    traindata, trainlabels = make_forest_input(trainfeatures)
    testdata, testlabels = make_forest_input(testfeatures)

    rf = Skrf()
    rf.fit(traindata, trainlabels)

    result = rf.predict(testdata)

    print result.shape
    print testlabels.shape

    result = zip(result, testlabels)

    # For testing of the random forest:
    # Take 50 true and 25 false paths for later testing
    # The remaining paths are for learning

    # random.seed()
    # testfeatures = IPL()
    # testfeatures['true'] = np.zeros((0, features['true'].shape[1]))
    # for i in xrange(0, 50):
    #     item = random.randrange(0, features['true'].shape[0], 1)
    #     # print item
    #     testfeatures['true'] = np.concatenate((testfeatures['true'], np.array([features['true'][item, :]])), axis=0)
    #     features['true'] = np.delete(features['true'], item, 0)
    #
    #     testfeatures['false'] = np.zeros((0, features['false'].shape[1]))
    # for i in xrange(0, 25):
    #     item = random.randrange(0, features['false'].shape[0], 1)
    #     # print item
    #     testfeatures['false'] = np.concatenate((testfeatures['false'], np.array([features['false'][item, :]])), axis=0)
    #     features['false'] = np.delete(features['false'], item, 0)
    #
    # traindata, trainlabels = make_forest_input(features)
    # testdata, testlabels = make_forest_input(testfeatures)
    #
    # rf = Skrf(oob_score=True)
    # rf.fit(traindata, trainlabels)
    # result = rf.predict(testdata)
    # features.logging('The result \n---\n{}', result)
    # features.logging('+++\nThe ground truth\n---\n{}', testlabels)

    return result


def random_forest_iteration(features):

    result = IPL()

    for k, v in features.iteritems():

        # Make the feature arrays
        v['0']['true'] = make_feature_array(v['0']['true'])
        v['0']['false'] = make_feature_array(v['0']['false'])
        v['1']['true'] = make_feature_array(v['1']['true'])
        v['1']['false'] = make_feature_array(v['1']['false'])

        result[k, '0'] = random_forest(v['0'], v['1'])
        result[k, '1'] = random_forest(v['1'], v['0'])

    return result


if __name__ == '__main__':

    resultsfolder = '/mnt/localdata02/jhennies/neuraldata/results/cremi_2016/161107_random_forest_of_paths/'

    yamlfile = resultsfolder + '/parameters.yml'

    features = IPL(
        yaml=yamlfile,
        yamlspec={'path': 'intermedfolder', 'filename': 'featurefile'}
    )
    params = features.get_params()
    thisparams = params['localmax_on_disttransf']
    features.startlogger(filename=params['resultfolder'] + 'random_forest.log', type='w')

    try:

        # Copy the script file and the parameters to the scriptsfolder
        copy(inspect.stack()[0][1], params['scriptsfolder'])
        copy(yamlfile, params['scriptsfolder'] + 'random_forest.parameters.yml')
        # Write script and parameters to the logfile
        features.code2log(inspect.stack()[0][1])
        features.logging('')
        features.yaml2log()
        features.logging('')

        features.logging('\nfeatures datastructure: \n---\n{}', features.datastructure2string(maxdepth=2))

        result = random_forest_iteration(features)

        result.write(filepath=params['intermedfolder'] + params['randforestfile'])

        features.logging('\nFinal dictionary structure:\n---\n{}', features.datastructure2string(maxdepth=2))
        features.logging('')
        features.stoplogger()

    except:
        raise
        features.errout('Unexpected error')

