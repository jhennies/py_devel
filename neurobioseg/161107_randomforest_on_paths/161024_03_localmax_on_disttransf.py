
from image_processing import ImageFileProcessing
from hdf5_image_processing import Hdf5ImageProcessingLib as IPL
import random
import vigra
import numpy as np
import os
import inspect
from shutil import copy

__author__ = 'jhennies'


def localmax_on_disttransf(ipl, keys, thisparams):

    locmaxnames = params['locmaxnames']
    dict_locmaxnames = dict(zip(keys, locmaxnames))
    for k in keys:
        ipl.rename_entry(old=k, new=dict_locmaxnames[k])

    # Gaussian smoothing
    ipl.logging('Dragging Carl Friedrich over the image ...')
    ipl.gaussian_smoothing(thisparams['sigma'] / np.array(thisparams['anisotropy']),
                           keys=locmaxnames)

    # Local maxima
    ipl.logging('Discovering mountains ...')
    ipl.extended_local_maxima(neighborhood=26, keys=locmaxnames)

    return ipl


def localmax_on_disttransf_image_iteration(ipl):

    params = ipl.get_params()
    thisparams = params['localmax_on_disttransf']

    for d, k, v, kl in ipl.data_iterator(yield_short_kl=True):

        if k == params['locmaxbordernames'][2]:
            ipl[kl].setlogger(ipl.getlogger())
            ipl[kl] = localmax_on_disttransf(ipl[kl], (params['locmaxbordernames'][2], params['locmaxbordernames'][3]), thisparams)


if __name__ == '__main__':

    resultsfolder = '/mnt/localdata02/jhennies/neuraldata/results/cremi_2016/161107_random_forest_of_paths/'

    yamlfile = resultsfolder + '/parameters.yml'

    ipl = IPL(
        yaml=yamlfile,
        yamlspec={'path': 'intermedfolder', 'filename': 'locmaxborderfile', 'skeys': {'locmaxbordernames': (2, 3)}},
        recursive_search=True
    )
    params = ipl.get_params()
    thisparams = params['localmax_on_disttransf']
    ipl.startlogger(filename=params['resultfolder'] + 'localmax_on_disttransf.log', type='w')

    try:

        # Copy the script file and the parameters to the scriptsfolder
        copy(inspect.stack()[0][1], params['scriptsfolder'])
        copy(yamlfile, params['scriptsfolder'] + 'localmax_on_disttransf.parameters.yml')
        # Write script and parameters to the logfile
        ipl.code2log(inspect.stack()[0][1])
        ipl.logging('')
        ipl.yaml2log()
        ipl.logging('')

        ipl.logging('\nipl datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

        localmax_on_disttransf_image_iteration(ipl)

        ipl.write(filepath=params['intermedfolder'] + params['locmaxfile'])

        ipl.logging('\nFinal dictionary structure:\n---\n{}', ipl.datastructure2string())
        ipl.logging('')
        ipl.stoplogger()

    except:

        ipl.errout('Unexpected error')
