
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
