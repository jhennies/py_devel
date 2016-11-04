
from image_processing import ImageFileProcessing
from hdf5_image_processing import Hdf5ImageProcessing, Hdf5ImageProcessingLib as IPL
from processing_lib import getlabel
import  processing_lib as lib
from hdf5_processing import Hdf5Processing
import random
import vigra
import numpy as np
import os
import vigra.graphs as graphs
import sys
import traceback
import inspect

__author__ = 'jhennies'


def make_path_images(path_processing, hfp, shape):

    path_processing['paths_true'] = np.zeros(shape, dtype=np.float32)

    # Make path image (true)
    c = 0
    # This iterates over the paths of class 'true'
    for d, k, v, kl in hfp.data_iterator(maxdepth=2, data=hfp['true']):
        if d == 2:
            if v.any():
                path = lib.swapaxes(v, 0, 1)
                path_processing.positions2value(path, c, keys='paths_true')

                c += 1

    path_processing['paths_false'] = np.zeros(shape, dtype=np.float32)

    # Make path image (false)
    c = 0
    # This iterates over the paths of class 'false'
    for d, k, v, kl in hfp.data_iterator(maxdepth=2, data=hfp['false']):
        if d == 2:
            if v.any():
                path = lib.swapaxes(v, 0, 1)
                path_processing.positions2value(path, c, keys='paths_false')

                c += 1


def make_features_paths(paths, disttransf_images, feature_images, features_out):

    paths.astype(np.uint32)
    feature_images.astype(np.float32)
    # features_out = Hdf5ImageProcessing()
    features_out['disttransf'] = IPL(data=paths)

    features_out['disttransf'].anytask(vigra.analysis.extractRegionFeatures, ignoreLabel=0,
                         reverse_order=True, reciprocal=False,
                         keys=('paths_true', 'paths_false'),
                         indict=disttransf_images)

    features_out['raw'] = paths
    features_out['raw'].anytask(vigra.analysis.extractRegionFeatures, ignoreLabel=0,
                                reverse_order=True, reciprocal=False,
                                keys=('paths_true', 'paths_false'),
                                indict=feature_images)


def features_along_paths(paths, feature_im, features_out):
    """
    :param paths: np.array containing the paths
    :param features: dict of np.arrays with the feature data
    :param features_out: the features along the paths
    :return:
    """
    feats = vigra.analysis.extractRegionFeatures(feature_im, paths, ignoreLabel=0)

    return feats


def get_features(paths, featureimage, featurelist, max_paths_per_label, hfp=None):

    newfeats = IPL()
    keylist = range(0, max_paths_per_label)
    keylist = [str(x) for x in keylist]
    for i, keys, vals in paths.simultaneous_iterator(
            max_count_per_item=max_paths_per_label,
            keylist=keylist):

        if hfp is not None:
            hfp.logging('Working in iteration = {}', i)

        image = np.zeros(featureimage.shape, dtype=np.uint32)

        c = 1
        for curk, curv in (dict(zip(keys, vals))).iteritems():

            curv = lib.swapaxes(curv, 0, 1)
            lib.positions2value(image, curv, c)
            c += 1

        newnewfeats = IPL(
            data=vigra.analysis.extractRegionFeatures(
                featureimage,
                image, ignoreLabel=0,
                features=featurelist
            )
        )

        for k, v in newnewfeats.iteritems():
            newnewfeats[k] = newnewfeats[k][1:]
            if k in newfeats:
                try:
                    newfeats[k] = np.concatenate((newfeats[k], newnewfeats[k]))
                except ValueError:
                    pass
            else:
                newfeats[k] = newnewfeats[k]

    return newfeats


def features_of_paths(hfp, disttransf_images, feature_images, features):

    # for d, k, v, kl in hfp.data_iterator():
    #
    #     if d == 3:
    #         print kl
    #
    #         im = np.zeros(disttransf_images['disttransf'].shape, dtype=np.uint32)
    #         v = lib.swapaxes(v, 0, 1)
    #         lib.positions2value(im, v, 1)
    #
    #         features[kl] = vigra.analysis.extractRegionFeatures(
    #             disttransf_images['disttransf'],
    #             im, ignoreLabel=0
    #         )

    params = hfp.get_params()
    thisparams = params['features_of_paths']

    # TODO: This contains redundant computation! The same path images are constructed several times

    for k, v in hfp['true'].iteritems():

        features['true', k, 'disttransf'] = get_features(
            v, disttransf_images['disttransf'],
            thisparams['features'],
            thisparams['max_paths_per_label'],
            hfp=hfp
        )

        for fk, fv in feature_images.iteritems():

            features['true', k, fk] = get_features(
                v, fv,
                thisparams['features'],
                thisparams['max_paths_per_label'],
                hfp=hfp
            )

    for k, v in hfp['false'].iteritems():

        features['false', k, 'disttransf'] = get_features(
            v, disttransf_images['disttransfm'],
            thisparams['features'],
            thisparams['max_paths_per_label'],
            hfp=hfp
        )

        for fk, fv in feature_images.iteritems():

            features['false', k, fk] = get_features(
                v, fv,
                thisparams['features'],
                thisparams['max_paths_per_label'],
                hfp=hfp
            )

    # for i, k, ks, vs in hfp['paths_true'].simultaneous_iterator():
    #     pass

if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'
    hfp = IPL(
        yaml=yamlfile,
        yamlspec={'path': 'intermedfolder', 'filename': 'pathstruefile'},
        castkey=None
    )
    # hfp.logging('datastructure:\n---\n{}', hfp.datastructure2string())
    params = hfp.get_params()

    hfp['true', 'border'] = IPL(data=hfp['largeobj', 'border_locmax', 'path'])
    hfp['true', 'locmax'] = IPL(data=hfp['largeobj', 'locmax', 'path'])
    hfp.pop('largeobj')

    hfp.data_from_file(filepath=params['intermedfolder'] + params['pathsfalsefile'])

    hfp['false', 'border'] = IPL(data=hfp['largeobjm', 'border_locmax_m', 'path'])
    hfp['false', 'locmax'] = IPL(data=hfp['largeobjm', 'locmaxm', 'path'])
    hfp.pop('largeobjm')

    hfp.pop('pathsim')
    hfp.pop('overlay')

    # # TODO: Insert code here
    # hfp = Hdf5ImageProcessingLib(
    #     yaml=yamlfile,
    #     yamlspec={'path': 'intermedfolder', 'filename': 'pathstruefile'},
    #     tkeys='true',
    #     castkey=None
    # )
    # params = hfp.get_params()
    # hfp.logging('params = {}', params)
    # hfp.data_from_file(
    #     filepath=params['intermedfolder'] + params['pathsfalsefile'],
    #     tkeys='false',
    #     castkey=None
    # )
    hfp.startlogger(filename=params['intermedfolder']+'features_of_paths.log', type='a')

    try:

        hfp.code2log(inspect.stack()[0][1])
        hfp.logging('')
        hfp.yaml2log()
        hfp.logging('')

        hfp.logging('\nhfp datastructure: \n\n{}', hfp.datastructure2string(maxdepth=1))

        # Done: Iterate over paths and accumulate features
        # Done: Implement data iterator

        # Done: Make path image (true)
        # Done: Make path image (false)
        # TODO: Get topological features
        # Done:     Topological feature: Length
        # TODO:     Topological feature: Statistics on curvature
        # TODO: Get data features on path (raw, probabilities, distance transform)
        # TODO: Get data features on end points (raw, probabilities, distance transform)
        # Done: Cross-computation of two ImageProcessing instances

        # Done: This is total bullshit! I need to iterate over all paths and extract the region features individually!

        # Store all feature images in here
        disttransf_images = IPL(
            yaml=yamlfile,
            yamlspec={'path': 'intermedfolder', 'filename': 'locmaxborderfile', 'skeys': {'locmaxbordernames': (2, 3)}},
            tkeys=('disttransf', 'disttransfm')
        )
        feature_images = IPL(
            yaml=yamlfile,
            yamlspec={'path': 'datafolder', 'filename': 'rawdatafile', 'skeys': 'rawdataname'},
            tkeys='raw'
        )
        feature_images.astype(np.float32)
        features = IPL()
        features_of_paths(hfp, disttransf_images, feature_images, features)



        # # feature_images.data_from_file(params[''])
        #
        # hfp.logging('\ndisttransf_images datastructure: \n\n{}', disttransf_images.datastructure2string(maxdepth=1))
        #
        # # This is for the path images
        # paths = IPL()
        # # # Add the feature images to the paths dictionary
        # # paths.set_data_dict(feature_images.get_data(), append=True)
        #
        # # Create the path images for feature accumulator
        # make_path_images(paths, hfp, disttransf_images['disttransf'].shape)
        # # paths.write(filepath='/media/julian/Daten/neuraldata/cremi_2016/test.h5')
        #
        # # This is for the features
        # features = IPL()
        #
        # # Get features along the paths
        # make_features_paths(paths, disttransf_images, feature_images, features)
        #
        # hfp.logging('\nCalculated features: \n-------------------\n{}-------------------\n', features.datastructure2string())

        # hfp.logging('Possible features: \n{}', features['paths_false', 'disttransfm'].supportedFeatures())
        features.write(filepath=params['intermedfolder'] + params['featurefile'])

        hfp.logging('')
        hfp.stoplogger()

    except ValueError:

        hfp.errout('Unexpected error', traceback)

