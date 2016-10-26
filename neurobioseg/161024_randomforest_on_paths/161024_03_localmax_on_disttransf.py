
from image_processing import ImageFileProcessing
from hdf5_image_processing import Hdf5ImageProcessingLib as IPL
import random
import vigra
import numpy as np
import os
import inspect
from shutil import copy

__author__ = 'jhennies'


def localmax_on_disttransf(hfp, keys):

    params = hfp.get_params()
    thisparams = params['localmax_on_disttransf']

    locmaxnames = params['locmaxnames']
    dict_locmaxnames = dict(zip(keys, locmaxnames))
    for k in keys:
        hfp.rename_entry(old=k, new=dict_locmaxnames[k])

    # Gaussian smoothing
    hfp.logging('Dragging Carl Friedrich over the image ...')
    hfp.gaussian_smoothing(thisparams['sigma'] / np.array(thisparams['anisotropy']),
                           keys=locmaxnames)

    # Local maxima
    hfp.logging('Discovering mountains ...')
    hfp.extended_local_maxima(neighborhood=26, keys=locmaxnames)


# if __name__ == '__main__':
#
#     yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'
#
#     ifp = ImageFileProcessing(
#         yaml=yamlfile,
#         yamlspec={'image_path': 'intermedfolder', 'image_file': 'largeobjfile', 'image_names': 'largeobjname'},
#         asdict=True,
#         keys=('largeobj',)
#     )
#     params = ifp.get_params()
#     thisparams = params['localmax_on_disttransf']
#     ifp.addfromfile(params['intermedfolder']+params['largeobjmfile'], image_names=params['largeobjmnames'][0], ids='largeobjm')
#
#     ifp.startlogger(filename=ifp.get_params()['intermedfolder'] + 'locmax_on_disttransf.log', type='a')
#
#     # ifp.code2log(__file__)
#     ifp.code2log(inspect.stack()[0][1])
#     ifp.logging('')
#
#     ifp.logging('yamlfile = {}', yamlfile)
#     ifp.logging('ifp.get_data().keys() = {}', ifp.get_data().keys())
#     ifp.logging('ifp.shape() = {}', ifp.shape())
#
#     # Boundary distance transform
#     # a) Boundaries
#     ifp.logging('Finding boundaries ...')
#     ifp.pixels_at_boundary(
#         axes=(np.array(thisparams['anisotropy']).astype(np.float32) ** -1).astype(np.uint8)
#     )
#     ifp.astype(np.float32)
#
#     # b) Distance transform
#     ifp.logging('Computing distance transform on boundaries ...')
#     ifp.distance_transform(
#         pixel_pitch=thisparams['anisotropy'],
#         background=True
#     )
#     locmaxnames = params['locmaxnames']
#     ifp.rename_entries(ids = ('largeobj', 'largeobjm'), targetids = (locmaxnames[0], locmaxnames[1]))
#
#     # Gaussian smoothing
#     ifp.logging('Dragging Carl Friedrich over the image ...')
#     ifp.gaussian_smoothing(thisparams['sigma'] / np.array(thisparams['anisotropy']),
#                            ids=(locmaxnames[0], locmaxnames[1]), targetids=(locmaxnames[2], locmaxnames[3]))
#
#     # Local maxima
#     ifp.logging('Discovering mountains ...')
#     ifp.extended_local_maxima(neighborhood=26, ids=(locmaxnames[2], locmaxnames[3]))
#
#     # Write result
#     # ifp.rename_entries(ids = ('largeobj', 'largeobjm'), targetids = params['locmaxnames'])
#     ifp.write(filename=params['locmaxfile'])
#
#     ifp.logging('')
#     ifp.stoplogger()


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'

    hfp = IPL(
        yaml=yamlfile,
        yamlspec={'path': 'intermedfolder', 'filename': 'locmaxborderfile', 'skeys': {'locmaxbordernames': (2, 3)}},
        tkeys=('disttransf', 'disttransfm'),
        castkey=None
    )
    params = hfp.get_params()
    thisparams = params['localmax_on_disttransf']
    hfp.startlogger(filename=params['resultfolder'] + 'localmax_on_disttransf.log', type='w')

    try:

        # Copy the script file and the parameters to the scriptsfolder
        copy(inspect.stack()[0][1], params['scriptsfolder'])
        copy(yamlfile, params['scriptsfolder'] + 'localmax_on_disttransf.parameters.yml')
        # Write script and parameters to the logfile
        hfp.code2log(inspect.stack()[0][1])
        hfp.logging('')
        hfp.yaml2log()
        hfp.logging('')

        hfp.logging('\nhfp datastructure: \n\n{}', hfp.datastructure2string(maxdepth=1))

        localmax_on_disttransf(hfp, ('disttransf', 'disttransfm'))

        hfp.write(filepath=params['intermedfolder'] + params['locmaxfile'])

        hfp.logging('\nFinal dictionary structure:\n---\n{}', hfp.datastructure2string())
        hfp.logging('')
        hfp.stoplogger()

    except:

        hfp.errout('Unexpected error')
