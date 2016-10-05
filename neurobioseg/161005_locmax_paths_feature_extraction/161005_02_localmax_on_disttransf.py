
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
    thisparams = params['localmax_on_disttransf']
    ifp.addfromfile(params['intermedfolder']+params['largeobjmfile'], image_names=params['largeobjmnames'][0], ids='largeobjm')

    ifp.startlogger(filename=ifp.get_params()['intermedfolder'] + 'locmax_on_disttransf.log', type='a')

    ifp.code2log(__file__)
    ifp.logging('')

    ifp.logging('yamlfile = {}', yamlfile)
    ifp.logging('ifp.get_data().keys() = {}', ifp.get_data().keys())
    ifp.logging('ifp.shape() = {}', ifp.shape())

    # Boundary distance transform
    # a) Boundaries
    ifp.logging('Finding boundaries ...')
    ifp.pixels_at_boundary(
        axes=(np.array(thisparams['anisotropy']).astype(np.float32) ** -1).astype(np.uint8)
    )
    ifp.astype(np.float32)

    # b) Distance transform
    ifp.logging('Computing distance transform on boundaries ...')
    ifp.distance_transform(
        pixel_pitch=thisparams['anisotropy'],
        background=True
    )
    locmaxnames = params['locmaxnames']
    ifp.rename_entries(ids = ('largeobj', 'largeobjm'), targetids = (locmaxnames[0], locmaxnames[1]))

    # Gaussian smoothing
    ifp.logging('Dragging Carl Friedrich over the image ...')
    ifp.gaussian_smoothing(thisparams['sigma'] / np.array(thisparams['anisotropy']),
                           ids=(locmaxnames[0], locmaxnames[1]), targetids=(locmaxnames[2], locmaxnames[3]))

    # Local maxima
    ifp.logging('Discovering mountains ...')
    ifp.extended_local_maxima(neighborhood=26, ids=(locmaxnames[2], locmaxnames[3]))

    # Write result
    # ifp.rename_entries(ids = ('largeobj', 'largeobjm'), targetids = params['locmaxnames'])
    ifp.write(filename=params['locmaxfile'])

    ifp.logging('')
    ifp.stoplogger()