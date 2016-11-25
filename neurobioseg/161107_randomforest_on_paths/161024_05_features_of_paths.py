
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


def make_path_images(path_processing, ipl, shape):

    path_processing['paths_true'] = np.zeros(shape, dtype=np.float32)

    # Make path image (true)
    c = 0
    # This iterates over the paths of class 'true'
    for d, k, v, kl in ipl.data_iterator(maxdepth=2, data=ipl['true']):
        if d == 2:
            if v.any():
                path = lib.swapaxes(v, 0, 1)
                path_processing.positions2value(path, c, keys='paths_true')

                c += 1

    path_processing['paths_false'] = np.zeros(shape, dtype=np.float32)

    # Make path image (false)
    c = 0
    # This iterates over the paths of class 'false'
    for d, k, v, kl in ipl.data_iterator(maxdepth=2, data=ipl['false']):
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


def get_features(paths, featureimage, featurelist, max_paths_per_label, ipl=None):

    newfeats = IPL()

    # TODO: Selection of a limited amount of paths should be random
    keylist = range(0, max_paths_per_label)
    keylist = [str(x) for x in keylist]

    # Iterate over all paths, yielding a list of one path per label object until no paths are left
    for i, keys, vals in paths.simultaneous_iterator(
            max_count_per_item=max_paths_per_label,
            keylist=keylist):

        if ipl is not None:
            ipl.logging('Working in iteration = {}', i)

        # Create a working image
        image = np.zeros(featureimage.shape, dtype=np.uint32)
        # And fill it with one path per label object
        c = 1
        for curk, curv in (dict(zip(keys, vals))).iteritems():
            curv = lib.swapaxes(curv, 0, 1)
            lib.positions2value(image, curv, c)
            c += 1

        # Extract the region features of the working image
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


def features_of_paths(ipl, disttransf_images, feature_images, thisparams):
    """
    The following datastructure is necessary for the dicts 'disttransf_images' and 'feature_images':
    true
    .   [locmax_name]
    .   .   [feature_name]
    false
    .   [locmax_name]
    .   .   [feature_name]

    ipl has this datastructure:
    true
    .   [locmax_name]
    .   .   [labels]
    .   .   .   [paths]
    false
    .   [locmax_name]
    .   .   [labels]
    .   .   .   [paths]

    :param ipl:
    :param disttransf_images:
    :param feature_images:
    :param thisparams:
    :return:
    """

    # TODO: This contains redundant computation! The same path images are constructed several times

    features = IPL()

    for k, v in ipl['true'].iteritems():

        ipl.logging('Key in class=true: {}', k)

        ipl.logging('Feature image: {}', 'disttransf')
        features['true', k, 'disttransf'] = get_features(
            v, disttransf_images['disttransf'],
            thisparams['features'],
            thisparams['max_paths_per_label_true'],
            ipl=ipl
        )

        for fk, fv in feature_images.iteritems():

            ipl.logging('Feature image: {}', fk)
            features['true', k, fk] = get_features(
                v, fv,
                thisparams['features'],
                thisparams['max_paths_per_label_true'],
                ipl=ipl
            )

    for k, v in ipl['false'].iteritems():

        ipl.logging('Key in class=false: {}', k)

        ipl.logging('Feature image: {}', 'disttransfm')
        features['false', k, 'disttransf'] = get_features(
            v, disttransf_images['disttransfm'],
            thisparams['features'],
            thisparams['max_paths_per_label_false'],
            ipl=ipl
        )

        for fk, fv in feature_images.iteritems():

            ipl.logging('Feature image: {}', fk)
            features['false', k, fk] = get_features(
                v, fv,
                thisparams['features'],
                thisparams['max_paths_per_label_false'],
                ipl=ipl
            )

    return features


def features_of_paths_image_iteration(ipl, disttransf_images, feature_images):

    params = ipl.get_params()
    thisparams = params['features_of_paths']

    features = IPL()

    for d, k, v, kl in ipl.data_iterator(yield_short_kl=True):

        if k == 'true':

            ipl.logging('Key list: {}', kl)

            ipl[kl].setlogger(ipl.getlogger())
            features[kl] = features_of_paths(
                ipl[kl], disttransf_images[kl], feature_images[kl],
                thisparams
            )

    return features


if __name__ == '__main__':

    resultsfolder = '/mnt/localdata02/jhennies/neuraldata/results/cremi_2016/161110_random_forest_of_paths/'

    yamlfile = resultsfolder + '/parameters.yml'

    ipl = IPL(
        yaml=yamlfile,
        yamlspec={'path': 'intermedfolder', 'filename': 'pathstruefile'},
        skeys='path', recursive_search=True
    )
    # ipl.logging('datastructure:\n---\n{}', ipl.datastructure2string())
    params = ipl.get_params()
    ipl.rename_layer('largeobj', 'true')

    # ipl['true', 'border'] = IPL(data=ipl['largeobj', 'border_locmax', 'path'])
    # ipl['true', 'locmax'] = IPL(data=ipl['largeobj', 'locmax', 'path'])
    # ipl.pop('largeobj')

    ipl.data_from_file(filepath=params['intermedfolder'] + params['pathsfalsefile'],
                       skeys='path', recursive_search=True, integrate=True)
    ipl.rename_layer('largeobjm', 'false')
    ipl.remove_layer('path')

    # ipl['false', 'border'] = IPL(data=ipl['largeobjm', 'border_locmax_m', 'path'])
    # ipl['false', 'locmax'] = IPL(data=ipl['largeobjm', 'locmaxm', 'path'])
    # ipl.pop('largeobjm')
    #
    # ipl.pop('pathsim')
    # ipl.pop('overlay')

    ipl.startlogger(filename=params['resultfolder']+'features_of_paths.log', type='w')

    try:

        ipl.code2log(inspect.stack()[0][1])
        ipl.logging('')
        ipl.yaml2log()
        ipl.logging('')

        ipl.logging('\nipl datastructure: \n\n{}', ipl.datastructure2string(maxdepth=4))

        # Done: Make path image (true)
        # Done: Make path image (false)
        # TODO: Get topological features
        # TODO:     Topological feature: Length (with respect to anisotropy!)
        # TODO:     Topological feature: Statistics on curvature
        # TODO: Get data features on path (raw, probabilities, distance transform)
        # TODO: Get data features on end points (raw, probabilities, distance transform)
        # Done: Cross-computation of two ImageProcessing instances

        # Done: This is total bullshit! I need to iterate over all paths and extract the region features individually!

        # Store all feature images in here
        disttransf_images = IPL(
            yaml=yamlfile,
            yamlspec={'path': 'intermedfolder', 'filename': 'locmaxborderfile', 'skeys': {'locmaxbordernames': (2, 3)}},
            recursive_search=True
        )
        feature_images = IPL(
            yaml=yamlfile,
            yamlspec={'path': 'datafolder', 'filename': 'rawdatafile', 'skeys': 'rawdataname'},
            recursive_search=True
        )
        ipl.logging('\nDisttransf images datastructure: \n---\n{}', disttransf_images.datastructure2string(maxdepth=4))
        ipl.logging('\nFeature images datastructure: \n---\n{}', feature_images.datastructure2string(maxdepth=4))
        feature_images.astype(np.float32)
        # features = IPL()
        features = features_of_paths_image_iteration(ipl, disttransf_images, feature_images)

        features.write(filepath=params['intermedfolder'] + params['featurefile'])

        ipl.logging('\nFinal datastructure:\n---\n{}', features.datastructure2string())

        ipl.logging('')
        ipl.stoplogger()

    except ValueError:

        ipl.errout('Unexpected error', traceback)

