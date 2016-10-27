
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


def shortest_paths(indicator, pairs, bounds=None):

    # Crate the grid graph and shortest path objects
    gridgr = graphs.gridGraph(indicator.shape)
    indicator = indicator.astype(np.float32)
    gridgr_edgeind = graphs.edgeFeaturesFromImage(gridgr, indicator)
    instance = graphs.ShortestPathPathDijkstra(gridgr)

    # Initialize paths image
    pathsim = np.zeros(indicator.shape)
    # Initialize list of path coordinates
    paths = []

    for pair in pairs:

        source = pair[0]
        target = pair[1]

        targetNode = gridgr.coordinateToNode(target)
        sourceNode = gridgr.coordinateToNode(source)

        instance.run(gridgr_edgeind, sourceNode, target=targetNode)
        path = instance.path(pathType='coordinates')
        # Do not forget to correct for the offset caused by cropping!
        if bounds is not None:
            paths.append(path + [bounds[0].start, bounds[1].start, bounds[2].start])
        else:
            paths.append(path)

        pathindices = np.swapaxes(path, 0, 1)
        pathsim[pathindices[0], pathindices[1], pathindices[2]] = 1

    return paths, pathsim


def find_shortest_path(hfp, penaltypower, bounds, disttransf, locmax):

    # TODO: distancetransform modifiactation can be done beforehand!
    # Modify distancetransform
    #
    # a) Invert: the lowest values (i.e. the lowest penalty for the shortest path detection) should be at the center of
    #    the current process
    disttransf = lib.invert_image(disttransf)
    # ifp.invert_image(ids='curdisttransf')
    #
    # b) Set all values outside the process to infinity
    disttransf = lib.filter_values(disttransf, np.amax(disttransf), type='eq', setto=np.inf)
    # ifp.filter_values(ifp.amax('curdisttransf'), type='eq', setto=np.inf, ids='curdisttransf', targetids='curdisttransf_inf')
    #
    # c) Increase the value difference between pixels near the boundaries and pixels central within the processes
    #    This increases the likelihood of the paths to follow the center of processes, thus avoiding short-cuts
    disttransf = lib.power(disttransf, penaltypower)
    # ifp.power(penaltypower, ids='curdisttransf_inf')

    # Get local maxima
    indices = np.where(locmax)
    coords = zip(indices[0], indices[1], indices[2])
    hfp.logging('Local maxima coordinates: {}', coords)

    # Make pairwise list of coordinates that will serve as source and target
    pairs = []
    for i in xrange(0, len(coords)-1):
        for j in xrange(i+1, len(coords)):
            pairs.append((coords[i], coords[j]))

    paths, pathim = shortest_paths(disttransf, pairs, bounds=bounds)

    return paths, pathim


def paths_within_labels(hfp, key, locmaxkeys, disttransfkey):

    params = hfp.get_params()
    thisparams = params['paths_within_labels']

    if type(locmaxkeys) is str:
        locmaxkeys = (locmaxkeys,)

    # This results in the format:
    # more_keys = (locmaxkeys[0], ..., locmaxkeys[n], disttransfkey)
    more_keys = locmaxkeys + (disttransfkey,)

    paths = IPL()
    for k in locmaxkeys:
        paths['pathsim', k] = np.zeros(hfp[key].shape)

    for lbl, lblim, more_ims, bounds in hfp.label_image_bounds_iterator(
        key=key, background=0, more_keys=more_keys,
        maskvalue=0, value=0
    ):
        # The format of more_ims is:
        # more_ims = {locmaxkeys[0]: locmax_1,
        #                ...
        #             locmaxkeys[n]: locmax_n,
        #             disttransfkey: disttransf}

        hfp.logging('======================\nWorking on label = {}', lbl)
        # hfp.logging('more_ims structure:\n---\n{}', more_ims.datastructure2string())
        hfp.logging('bounds = {}', bounds)

        for i in xrange(0, len(locmaxkeys)):

            hfp.logging('locmaxkey = {}', locmaxkeys[i])
            if np.amax(more_ims[locmaxkeys[i]]) > 0:
                ps, pathsim = find_shortest_path(hfp, thisparams['penaltypower'], bounds,
                                                 more_ims[disttransfkey], more_ims[locmaxkeys[i]])
                hfp.logging('Number of paths found: {}', len(ps))

                paths[key, locmaxkeys[i], 'path'] = ps
                paths[key, locmaxkeys[i], 'pathsim'] = pathsim

                paths['pathsim', locmaxkeys[i]][bounds][pathsim > 0] = pathsim[pathsim > 0]

    for k in locmaxkeys:
        paths['overlay', k] = np.array([paths['pathsim', k],
                                     hfp[key].astype(np.float32) / np.amax(hfp[key]),
                                     vigra.filters.multiBinaryDilation(hfp[k].astype(np.uint8), 5)])

    return paths


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'

    hfp = IPL(
        yaml=yamlfile,
        yamlspec={'path': 'intermedfolder', 'filename': 'locmaxborderfile', 'skeys': {'locmaxbordernames': (0, 2)}},
        tkeys=('border_locmax', 'disttransf'),
        castkey=None
    )
    params = hfp.get_params()
    thisparams = params['paths_within_labels']
    hfp.startlogger(filename=params['resultfolder'] + 'paths_within_labels.log', type='w')
    hfp.data_from_file(params['intermedfolder'] + params['largeobjfile'],
                       skeys=params['largeobjname'],
                       tkeys='largeobj')
    hfp.data_from_file(params['intermedfolder'] + params['locmaxfile'],
                       skeys=params['locmaxnames'][0],
                       tkeys='locmax')

    try:

        # Copy the script file and the parameters to the scriptsfolder
        copy(inspect.stack()[0][1], params['scriptsfolder'])
        copy(yamlfile, params['scriptsfolder'] + 'paths_within_labels.parameters.yml')
        # Write script and parameters to the logfile
        hfp.code2log(inspect.stack()[0][1])
        hfp.logging('')
        hfp.yaml2log()
        hfp.logging('')

        hfp.logging('\nhfp datastructure: \n\n{}', hfp.datastructure2string(maxdepth=1))

        paths = paths_within_labels(hfp, 'largeobj', ('border_locmax', 'locmax'), 'disttransf')

        paths.write(filepath=params['intermedfolder'] + params['pathstruefile'])

        hfp.logging('\nFinal dictionary structure:\n---\n{}', hfp.datastructure2string())
        hfp.logging('')
        hfp.stoplogger()

    except:

        hfp.errout('Unexpected error')
