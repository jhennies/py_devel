
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


def load_images(filepath, skeys=None, recursive_search=False, logger=None):

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


def features_of_paths(yparams):

    params = yparams.get_params()
    thisparams = rdict(params['features_of_paths'])

    featureims = ipl()

    # Load feature images
    feature_sources = thisparams['sources', 'featureims']
    feature_skwargs = thisparams['skwargs', 'featureims']
    for sourcekey, source in feature_sources.iteritems():

        # Load the necessary images
        #   1. Determine the settings for fetching the data
        try:
            recursive_search = False
            recursive_search = feature_skwargs['default', 'recursive_search']
            recursive_search = feature_skwargs[sourcekey, 'recursive_search']
        except KeyError:
            pass
        if len(source) > 2:
            skeys = source[2]
        else:
            skeys = None

        #   2. Load the data
        yparams.logging('skeys = {}', skeys)
        yparams.logging('recursive_search = {}', recursive_search)
        featureims[sourcekey] = load_images(
            params[source[0]] + params[source[1]], skeys=skeys, recursive_search=recursive_search,
            logger=yparams
        )

    yparams.logging('\nFeatureims datastructure: \n\n{}', featureims.datastructure2string())

    # Load true and false paths
    paths = ipl()
    paths_sources = thisparams['sources', 'paths']
    paths_skwargs = thisparams['skwargs', 'paths']
    for sourcekey, source in paths_sources.iteritems():

        # Load the necessary images
        #   1. Determine the settings for fetching the data
        try:
            recursive_search = False
            recursive_search = paths_skwargs['default', 'recursive_search']
            recursive_search = paths_skwargs[sourcekey, 'recursive_search']
        except KeyError:
            pass
        if len(source) > 2:
            skeys = source[2]
        else:
            skeys = None

        #   2. Load the data
        yparams.logging('skeys = {}', skeys)
        yparams.logging('recursive_search = {}', recursive_search)
        paths[sourcekey] = load_images(
            params[source[0]] + params[source[1]], skeys=skeys, recursive_search=recursive_search,
            logger=yparams
        )

    yparams.logging('\nPaths datastructure: \n\n{}', paths.datastructure2string(maxdepth=4))

    # Load the segmentation image datastructure (We just require the datastructure, not the data
    # itself)
    try:
        recursive_search = False
        recursive_search = thisparams['skwargs', 'segmentation', 'recursive_search']
    except KeyError:
        pass
    if len(thisparams['sources', 'segmentation']) > 2:
        skeys = thisparams['sources', 'segmentation'][2]
    else:
        skeys = None
    segmentation = load_images(
        params[thisparams['sources', 'segmentation'][0]] + params[thisparams['sources', 'segmentation'][1]],
        skeys=skeys, recursive_search=recursive_search,
        logger=yparams
    )

    yparams.logging('\nSegmentation datastructure: \n\n{}', segmentation.datastructure2string(maxdepth=4))

    # data['contacts'].reduce_from_leafs(iterate=True)
    # data['disttransf'].reduce_from_leafs(iterate=True)

    # Set targetfile
    featuresfile = params[thisparams['target'][0]] \
                 + params[thisparams['target'][1]]
    pathlistfile = params[thisparams['pathlist'][0]] \
                 + params[thisparams['pathlist'][1]]

    pathlist = ipl()

    for d, k, v, kl in segmentation.data_iterator(yield_short_kl=True, leaves_only=True):
        yparams.logging('===============================\nWorking on image: {}', kl + [k])

        # # TODO: Implement copy full logger
        # data[kl].set_logger(data.get_logger())

        # Bild an input featureims dict for the path computation
        infeatims = ipl()
        sourcelist = thisparams['sources', 'featureims'].dcp()
        if 'segmentation' in sourcelist:
            infeatims['segmentation'] = featureims['segmentation'][kl][k]
            sourcelist.pop('segmentation')
        for source in sourcelist:
            infeatims[source] = featureims[source][kl]
        infeatims.populate()

        # Bild an input dict for true paths
        intruepaths = paths['truepaths'][kl][k]['truepaths']
        infalsepaths = paths['falsepaths'][kl][k]['falsepaths']
        intruepaths.populate()
        infalsepaths.populate()

        yparams.logging('\nInfeatims datastructure: \n\n{}', infeatims.datastructure2string())
        yparams.logging('\nIntruepaths datastructure: \n\n{}', intruepaths.datastructure2string(maxdepth=3))
        yparams.logging('\nInfalsepaths datastructure: \n\n{}', infalsepaths.datastructure2string(maxdepth=3))


        features = ipl()
        features[kl + [k]], pathlist[kl + [k]] = libip.features_of_paths(
            yparams,
            intruepaths, infalsepaths,
            infeatims, infeatims, kl,
            return_pathlist=True
        )

        yparams.logging('\nPathlist datastructure: \n\n{}', pathlist.datastructure2string(function=type, leaves_only=False))

        # Write the result to file
        features.write(filepath=featuresfile)
        # pathlist.astype(np.uint8)
        # pathlist.write(filepath=pathlistfile)

    with open(pathlistfile, 'wb') as f:
        pickle.dump(pathlist, f)


def run_features_of_paths(yamlfile, logging=True):

    yparams = YamlParams(filename=yamlfile)
    params = yparams.get_params()

    # Logger stuff
    yparams.set_indent(1)
    yparams.startlogger(
        filename=params['resultfolder'] + 'features_of_paths.log',
        type='w', name='FeaturesOfPaths'
    )

    try:

        features_of_paths(yparams)

        yparams.logging('')
        yparams.stoplogger()

    except:

        yparams.errout('Unexpected error')


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters_ref.yml'

    run_features_of_paths(yamlfile, logging=False)