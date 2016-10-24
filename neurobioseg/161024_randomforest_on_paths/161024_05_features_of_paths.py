
from image_processing import ImageFileProcessing
from hdf5_image_processing import Hdf5ImageProcessing, Hdf5ImageProcessingLib
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
    for d in hfp.data_iterator(maxdepth=1, data=hfp['true']):
        if d['depth'] == 1:
            path = lib.swapaxes(d['val'], 0, 1)
            path_processing.positions2value(path, c)

            c += 1

        path_processing['paths_false'] = np.zeros(shape, dtype=np.float32)

    # Make path image (false)
    c = 0
    # This iterates over the paths of class 'false'
    for d in hfp.data_iterator(maxdepth=1, data=hfp['false']):
        if d['depth'] == 1:
            path = lib.swapaxes(d['val'], 0, 1)
            path_processing.positions2value(path, c)

            c += 1


def make_features_paths(paths, disttransf_images, feature_images, features_out):

    paths.astype(np.uint32)
    feature_images.astype(np.float32)
    # features_out = Hdf5ImageProcessing()
    features_out['disttransf'] = paths

    features_out['disttransf'].anytask(vigra.analysis.extractRegionFeatures, ignoreLabel=0,
                         reverse_order=True, reciprocal=False,
                         keys=('paths_true', 'paths_false'),
                         indict=disttransf_images)

if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'

    # TODO: Insert code here
    hfp = Hdf5ImageProcessingLib(
        yaml=yamlfile,
        yamlspec={'path': 'intermedfolder', 'filename': 'pathstruefile'},
        tkeys='true',
        castkey=None
    )
    params = hfp.get_params()
    hfp.logging('params = {}', params)
    hfp.data_from_file(
        filepath=params['intermedfolder'] + params['pathsfalsefile'],
        tkeys='false',
        castkey=None
    )
    hfp.startlogger(filename=params['intermedfolder']+'features_of_paths.log', type='a')

    try:

        hfp.code2log(inspect.stack()[0][1])
        hfp.logging('')
        hfp.yaml2log()
        hfp.logging('')

        hfp.logging('\nhfp datastructure: \n\n{}', hfp.datastructure2string(maxdepth=1))

        # TODO: Iterate over paths and accumulate features
        # TODO: Implement data iterator

        # ifp = ImageFileProcessing()

        # pathclass = None

        # for d in hfp.data_iterator(maxdepth=2):
        #
        #     if d['depth'] == 0:
        #         pathclass = d['key']
        #
        #     if d['depth'] == 2:
        #
        #         if len(d['val']) > 0:
        #
        #             hfp.logging('---')
        #             # hfp.logging('Class: {}', pathclass)
        #             # hfp.logging('From {} to {}', d['val'][0], d['val'][-1])
        #
        #             # TODO: Extract features of a path:
        #
        #             # Path length:
        #             pathlength = d['val'].shape[0]
        #             hfp.logging('pathlength = {}', pathlength)
        #
        #             # Coordinates of path:
        #             pathcoords = d['val']
        #             # Features arounds path
        #
        #             # Coordinates around ends:
        #             # Features around ends

        # Done: Make path image (true)
        # Done: Make path image (false)
        # TODO: Get topological features
        # TODO:     Topological feature: Length
        # TODO:     Topological feature: Statistics on curvature
        # TODO: Get data features on path (raw, probabilities, distance transform)
        # TODO: Get data features on end points (raw, probabilities, distance transform)
        # TODO: Cross-computation of two ImageProcessing instances

        # Store all feature images in here
        disttransf_images = Hdf5ImageProcessing(
            yaml=yamlfile,
            yamlspec={'path': 'intermedfolder', 'filename': 'locmaxfile', 'skeys': {'locmaxnames': (0, 1)}},
            tkeys=('disttransf', 'disttransfm')
        )
        feature_images = Hdf5ImageProcessing()

        hfp.logging('\ndisttransf_images datastructure: \n\n{}', disttransf_images.datastructure2string(maxdepth=1))

        # This is for the path images
        paths = Hdf5ImageProcessingLib()
        # # Add the feature images to the paths dictionary
        # paths.set_data_dict(feature_images.get_data(), append=True)

        # Create the path images for feature accumulator
        make_path_images(paths, hfp, disttransf_images['disttransf'].shape)
        # paths.write(filepath='/media/julian/Daten/neuraldata/cremi_2016/test.h5')

        # This is for the features
        features = Hdf5ImageProcessing()

        # Get features along the paths
        make_features_paths(paths, disttransf_images, feature_images, features)

        hfp.logging('\nCalculated features: \n-------------------\n{}-------------------\n', features.datastructure2string())

        # hfp.logging('Possible features: \n{}', features['paths_false', 'disttransfm'].supportedFeatures())
        # features.write(filepath='/media/julian/Daten/neuraldata/cremi_2016/test.h5')

        hfp.logging('')
        hfp.stoplogger()

    except:

        hfp.errout('Unexpected error', traceback)

