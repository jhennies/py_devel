
from image_processing import ImageFileProcessing
from hdf5_processing import Hdf5Processing
from hdf5_image_processing import Hdf5ImageProcessingLib as IPL
import random
import vigra
import numpy as np
import os
import vigra.graphs as graphs
import inspect
from shutil import copy
import processing_lib as lib

__author__ = 'jhennies'


def find_shortest_path(ipl, penaltypower, bounds, disttransf, locmax):

    # Modify distancetransform
    #
    # a) Invert: the lowest values (i.e. the lowest penalty for the shortest path detection) should be at the center of
    #    the current process
    disttransf = lib.invert_image(disttransf)
    #
    # b) Set all values outside the process to infinity
    disttransf = lib.filter_values(disttransf, np.amax(disttransf), type='eq', setto=np.inf)
    #
    # c) Increase the value difference between pixels near the boundaries and pixels central within the processes
    #    This increases the likelihood of the paths to follow the center of processes, thus avoiding short-cuts
    disttransf = lib.power(disttransf, penaltypower)

    # Get local maxima
    indices = np.where(locmax)
    coords = zip(indices[0], indices[1], indices[2])
    ipl.logging('Local maxima coordinates: {}', coords)

    # Make pairwise list of coordinates that will serve as source and target
    pairs = []
    for i in xrange(0, len(coords)-1):
        for j in xrange(i+1, len(coords)):
            pairs.append((coords[i], coords[j]))

    paths, pathim = lib.shortest_paths(disttransf, pairs, bounds=bounds, hfp=ipl)

    # # Make sure no empty paths lists are returned
    # paths = [x for x in paths if x.any()]
    return paths, pathim


def paths_within_labels(ipl, key, locmaxkeys, disttransfkey, thisparams, ignore=[]):

    if type(locmaxkeys) is str:
        locmaxkeys = (locmaxkeys,)

    # This results in the format:
    # more_keys = (locmaxkeys[0], ..., locmaxkeys[n], disttransfkey)
    more_keys = locmaxkeys + (disttransfkey,)

    paths = IPL()
    for k in locmaxkeys:
        paths['pathsim', k] = np.zeros(ipl[key].shape)

    for lbl, lblim, more_ims, bounds in ipl.label_image_bounds_iterator(
        key=key, background=0, more_keys=more_keys,
        maskvalue=0, value=0
    ):
        if lbl in ignore:
            continue
        # The format of more_ims is:
        # more_ims = {locmaxkeys[0]: locmax_1,
        #                ...
        #             locmaxkeys[n]: locmax_n,
        #             disttransfkey: disttransf}

        ipl.logging('======================\nWorking on label = {}', lbl)
        # ipl.logging('more_ims structure:\n---\n{}', more_ims.datastructure2string())
        ipl.logging('bounds = {}', bounds)

        for i in xrange(0, len(locmaxkeys)):

            ipl.logging('locmaxkey = {}', locmaxkeys[i])
            if np.amax(more_ims[locmaxkeys[i]]) > 0:
                ps, pathsim = find_shortest_path(ipl, thisparams['penaltypower'], bounds,
                                                 more_ims[disttransfkey], more_ims[locmaxkeys[i]])

                # Only store the path if the path-calculation successfully determined a path
                # Otherwise an empty list would be stored
                if ps:
                    ipl.logging('Number of paths found: {}', len(ps))

                    paths[key, locmaxkeys[i], 'path', lbl] = ps
                    paths[key, locmaxkeys[i], 'pathsim'] = pathsim

                    paths['pathsim', locmaxkeys[i]][bounds][pathsim > 0] = pathsim[pathsim > 0]

    for k in locmaxkeys:
        paths['overlay', k] = np.array([paths['pathsim', k],
                                     ipl[key].astype(np.float32) / np.amax(ipl[key]),
                                     vigra.filters.multiBinaryDilation(ipl[k].astype(np.uint8), 5)])

    return paths


def paths_within_labels_image_iteration(ipl):

    params = ipl.get_params()
    thisparams = params['paths_within_labels']

    paths = IPL()

    for d, k, v, kl in ipl.data_iterator(yield_short_kl=True):

        if k == params['locmaxbordernames'][0]:

            paths[kl] = paths_within_labels(ipl[kl], params['largeobjname'],
                                (params['locmaxbordernames'][0], params['locmaxnames'][0]),
                                params['locmaxbordernames'][2],
                                thisparams, ignore=thisparams['ignore'])

    return paths


if __name__ == '__main__':

    resultsfolder = '/mnt/localdata02/jhennies/neuraldata/results/cremi_2016/161107_random_forest_of_paths/'

    yamlfile = resultsfolder + '/parameters.yml'

    ipl = IPL(
        yaml=yamlfile,
        yamlspec={'path': 'intermedfolder', 'filename': 'locmaxborderfile', 'skeys': {'locmaxbordernames': (0, 2)}},
        recursive_search=True
    )
    params = ipl.get_params()
    thisparams = params['paths_within_labels']
    ipl.startlogger(filename=params['resultfolder'] + 'paths_within_labels.log', type='w')
    ipl.data_from_file(params['intermedfolder'] + params['largeobjfile'],
                       skeys=params['largeobjname'],
                       recursive_search=True,
                       integrate=True)
    ipl.data_from_file(params['intermedfolder'] + params['locmaxfile'],
                       skeys=params['locmaxnames'][0],
                       recursive_search=True,
                       integrate=True)

    try:

        # Copy the script file and the parameters to the scriptsfolder
        copy(inspect.stack()[0][1], params['scriptsfolder'])
        copy(yamlfile, params['scriptsfolder'] + 'paths_within_labels.parameters.yml')
        # Write script and parameters to the logfile
        ipl.code2log(inspect.stack()[0][1])
        ipl.logging('')
        ipl.yaml2log()
        ipl.logging('')

        ipl.logging('\nipl datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

        paths = paths_within_labels_image_iteration(ipl)

        paths.write(filepath=params['intermedfolder'] + params['pathstruefile'])

        ipl.logging('\nFinal dictionary structure:\n---\n{}', ipl.datastructure2string())
        ipl.logging('')
        ipl.stoplogger()

    except:

        ipl.errout('Unexpected error')
