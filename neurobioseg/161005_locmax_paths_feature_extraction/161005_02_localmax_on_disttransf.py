
from image_processing import ImageFileProcessing
import random
import vigra
import numpy as np
import os

__author__ = 'jhennies'




if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'

    ifp = ImageFileProcessing(
        yaml=yamlfile,
        yamlspec={'image_path': 'intermedfolder', 'image_file': 'largeobjfile', 'image_names': 'largeobjname'},
        asdict=True,
        keys=('largeobj',)
    )
    params = ifp.get_params()
    ifp.addfromfile(params['intermedfolder']+params['largeobjmfile'], image_names=params['largeobjmnames'][0], ids='largeobjm')

    ifp.startlogger(filename=ifp.get_params()['intermedfolder'] + 'locmax_on_disttransf.log', type='a')

    ifp.code2log(__file__)
    ifp.logging('')

    ifp.logging('yamlfile = {}', yamlfile)
    ifp.logging('ifp.get_data().keys() = {}', ifp.get_data().keys())
    ifp.logging('ifp.shape() = {}', ifp.shape())

    # Done: Boundary distance transform
    # Done:     Boundaries
    ifp.pixels_at_boundary(axes=(np.array(params['localmax_on_disttransf']['anisotropy']).astype(np.float32) ** -1).astype(np.uint8), ids='largeobj', targetids='largeobjboundaries')
    ifp.astype(np.float32, ids='largeobjboundaries')

    # Done:     Distance transform
    ifp.distance_transform(pixel_pitch=params['localmax_on_disttransf']['anisotropy'], background=True, ids='largeobjboundaries', targetids='disttransf')
    ifp.astype(np.uint8, ids='largeobjboundaries')
    ifp.write(filename = 'test.h5')

    # TODO: Gaussian smoothing
    # TODO: Local maxima

    ifp.logging('')
    ifp.stoplogger()