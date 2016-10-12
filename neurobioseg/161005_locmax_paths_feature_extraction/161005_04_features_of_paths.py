
from image_processing import ImageFileProcessing
from image_processing import getlabel
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


if __name__ == '__main__':

    try:

        yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'

        # ifp = ImageFileProcessing(
        #     yaml=yamlfile,
        #     yamlspec={'image_path': 'intermedfolder', 'image_file': 'locmaxfile', 'image_names': ('locmaxnames', 0, 1)},
        #     asdict=True,
        #     keys=('disttransf', 'disttransfm')
        # )
        # params = ifp.get_params()
        # thisparams = params['paths_of_partners']
        #
        # ifp.startlogger(filename=params['intermedfolder'] + 'features_of_paths.log', type='a')
        # ifp.code2log(inspect.stack()[0][1])
        # ifp.logging('')
        # ifp.yaml2log()
        # ifp.logging('')
        #
        # ifp.logging('yamlfile = {}', yamlfile)
        # ifp.logging('ifp.get_data().keys() = {}', ifp.get_data().keys())
        # ifp.logging('ifp.shape() = {}', ifp.shape())
        # ifp.logging('ifp.amax() = {}', ifp.amax())

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

        pathclass = None

        for d in hfp.data_iterator(maxdepth=2):

            if d['depth'] == 0:
                pathclass = d['key']

            if d['depth'] == 2:

                if len(d['val']) > 0:

                    hfp.logging('---')
                    # hfp.logging('Class: {}', pathclass)
                    # hfp.logging('From {} to {}', d['val'][0], d['val'][-1])

                    # TODO: Extract features of a path:

                    # Path length:
                    pathlength = d['val'].shape[0]
                    hfp.logging('pathlength = {}', pathlength)

                    # Coordinates of path:
                    pathcoords = d['val']
                    # Features arounds path

                    # Coordinates around ends:
                    # Features around ends

        # TODO: Make path image (true)
        # TODO: Make path image (false)
        # TODO: Get topological features
        # TODO:     Topological feature: Length
        # TODO:     Topological feature: Statistics on curvature
        # TODO: Get data features on path (raw, probabilities, distance transform)
        # TODO: Get data features on end points (raw, probabilities, distance transform)

        # Make path image
        feature_images = ImageFileProcessing(
            yaml=yamlfile,
            yamlspec={'image_path': 'intermedfolder', 'image_file': 'locmaxfile', 'image_names': ('locmaxnames', 0, 1)},
            asdict=True,
            keys=('disttransf', 'disttransfm')
        )
        path_processing = ImageFileProcessing()
        path_processing.new_image(feature_images.shape('disttransf'), 'paths_true', np.float32, 0)

        hfp.logging('path_processing.keys() = {}', path_processing.get_data().keys())

        hfp.logging('')
        hfp.stoplogger()

    except:

        hfp.errout('Unexpected error', traceback)

