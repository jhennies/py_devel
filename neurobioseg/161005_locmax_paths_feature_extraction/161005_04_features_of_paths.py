
from image_processing import ImageFileProcessing
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


def make_path_images(path_processing, hfp):

    path_processing.new_image(feature_images.shape('disttransf'), 'paths_true', np.float32, 0)

    # Make path image (true)
    c = 0
    # This iterates over the paths of class 'true'
    for d in hfp.data_iterator(maxdepth=1, data=hfp.getdata()['true']):
        if d['depth'] == 1:
            path = lib.swapaxes(d['val'], 0, 1)
            path_processing.positions2value(path, c)

            c += 1

    path_processing.new_image(feature_images.shape('disttransf'), 'paths_false', np.float32, 0)

    # Make path image (false)
    c = 0
    # This iterates over the paths of class 'false'
    for d in hfp.data_iterator(maxdepth=1, data=hfp.getdata()['false']):
        if d['depth'] == 1:
            path = lib.swapaxes(d['val'], 0, 1)
            path_processing.positions2value(path, c)

            c += 1


if __name__ == '__main__':

    try:

        yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'

        # TODO: Insert code here
        hfp = Hdf5Processing(
            yaml=yamlfile,
            yamlspec={'path': 'intermedfolder', 'filename': 'pathstruefile'},
            dataname='true',
            castkey=None
        )
        params = hfp.get_params()
        hfp.logging('params = {}', params)
        hfp.data_from_file(
            filepath=params['intermedfolder'] + params['pathsfalsefile'],
            dataname='false',
            castkey=None
        )
        hfp.startlogger(filename=params['intermedfolder']+'features_of_paths.log', type='a')
        hfp.code2log(inspect.stack()[0][1])
        hfp.logging('')
        hfp.yaml2log()
        hfp.logging('')

        hfp.logging('\nhfp datastructure: \n\n{}', hfp.datastructure2string(maxdepth=1))

        # TODO: Iterate over paths and accumulate features
        # TODO: Implement data itarator

        ifp = ImageFileProcessing()

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

        # Store all feature images in here
        feature_images = ImageFileProcessing(
            yaml=yamlfile,
            yamlspec={'image_path': 'intermedfolder', 'image_file': 'locmaxfile', 'image_names': ('locmaxnames', 0, 1)},
            asdict=True,
            keys=('disttransf', 'disttransfm')
        )

        # This is for the path images
        paths = ImageFileProcessing()

        # Create the path images for feature accumulator
        make_path_images(paths, hfp)

        paths.write(filepath='/media/julian/Daten/neuraldata/cremi_2016/test.h5')

        hfp.logging('')
        hfp.stoplogger()

    except:

        hfp.errout('Unexpected error', traceback)

