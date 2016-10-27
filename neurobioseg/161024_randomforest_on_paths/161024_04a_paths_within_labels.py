
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


def find_shortest_path(hfp, penaltypower, bounds, disttransf, locmax):

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

    indicator = disttransf
    # indicator = ifp.get_image('curdisttransf_inf')
    gridgr = graphs.gridGraph(disttransf.shape)

    hfp.logging('gridgr.shape = {}'.format(gridgr.shape))

    indicator = indicator.astype(np.float32)
    gridgr_edgeind = graphs.edgeFeaturesFromImage(gridgr, indicator)
    instance = graphs.ShortestPathPathDijkstra(gridgr)

    # Get two local maxima
    indices = np.where(locmax)
    # indices = np.where(ifp.get_image('curlocmax') == 1)
    coords = zip(indices[0], indices[1], indices[2])
    hfp.logging('Local maxima coordinates: {}', coords)

    # ifp.set_data_dict({'pathsim': np.zeros(ifp.get_image('curlocmax').shape)}, append=True)
    pathsim = np.zeros(locmax.shape)

    hfp.logging('len(coords) = {}', len(coords))
    paths = []
    for i in xrange(0, len(coords)-1):

        for j in xrange(i+1, len(coords)):

            hfp.logging('---')
            hfp.logging('i = {}; j = {}', i, j)

            source = coords[i]
            target = coords[j]

            targetNode = gridgr.coordinateToNode(target)
            sourceNode = gridgr.coordinateToNode(source)

            hfp.logging('Source = {}', source)
            hfp.logging('Target = {}', target)

            instance.run(gridgr_edgeind, sourceNode, target=targetNode)
            path = instance.path(pathType='coordinates')
            # Do not forget to correct for the offset caused by cropping!
            paths.append(path + [bounds[0].start, bounds[1].start, bounds[2].start])

            # for coords in path:
            #     # ifp.logging('coords = {}'.format(coords))
            #     pass

            pathindices = np.swapaxes(path, 0, 1)
            pathsim[pathindices[0], pathindices[1], pathindices[2]] = 1

    # # ifp.concatenate('disttransf', 'paths', target='paths_over_dist')
    # hfp.astype(np.uint8, ('pathsim', 'curlocmax'))
    # # ifp.anytask(vigra.filters.multiBinaryDilation, ('paths', 'locmax'), 3)
    # hfp.swapaxes(0, 2, ids=('pathsim', 'curlocmax', 'curdisttransf'))
    # hfp.anytask(vigra.filters.discDilation, 2, ids=('pathsim', 'curlocmax'))
    # hfp.set_data_dict({'paths_over_dist': np.array([ifp.get_image('pathsim'), ifp.get_image('curlocmax'), ifp.get_image('curdisttransf')])}, append=True)

    return paths, pathsim


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
            if np.amax(more_ims[locmaxkeys[i]]) == 1:
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


# if False:
#
#     yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'
#
#     ifp = ImageFileProcessing(
#         yaml=yamlfile,
#         yamlspec={'image_path': 'intermedfolder', 'image_file': 'locmaxfile', 'image_names': ('locmaxnames', 0, 2)},
#         asdict=True,
#         keys=('disttransf', 'locmax')
#     )
#     params = ifp.get_params()
#     thisparams = params['paths_within_labels']
#     ifp.addfromfile(params['intermedfolder']+params['largeobjfile'], image_names=params['largeobjname'], ids='largeobj')
#
#     ifp.startlogger(filename=params['intermedfolder'] + 'paths_within_labels.log', type='a')
#
#     # ifp.code2log(__file__)
#     ifp.code2log(inspect.stack()[0][1])
#     ifp.logging('')
#
#     ifp.logging('yamlfile = {}', yamlfile)
#     ifp.logging('ifp.get_data().keys() = {}', ifp.get_data().keys())
#     ifp.logging('ifp.shape() = {}', ifp.shape())
#     ifp.logging('{}', ifp.amax())
#
#     hfp = Hdf5Processing()
#     c = 0
#     for lblo in ifp.label_bounds_iterator('largeobj', 'curlabel', ids=('locmax', 'disttransf'), targetids=('curlocmax', 'curdisttransf'),
#                                           maskvalue=0, value=0, background=0):
#
#         ifp.logging('------------\nCurrent label {} in iteration {}', lblo['label'], c)
#         ifp.logging('Bounding box = {}', lblo['bounds'])
#
#         if ifp.amax('curlocmax') == 1:
#             ps = find_shortest_path(ifp, thisparams['penaltypower'], lblo['bounds'])
#             ifp.logging('Number of paths found: {}', len(ps))
#             if ps:
#                 hfp.setdata({lblo['label']: ps}, append=True)
#                 ifp.write(filename='paths_over_dist_true_{}.h5'.format(lblo['label']), ids='paths_over_dist')
#
#         else:
#             ifp.logging('No local maxima found for this label.')
#
#         c += 1
#         # if c == 5:
#         #     break
#
#     hfp.write(filepath=params['intermedfolder']+params['pathstruefile'])
#
#     ifp.logging('')
#     ifp.stoplogger()


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
        # hfp.write(filepath=params['intermedfolder'] + params['locmaxfile'])

        hfp.logging('\nFinal dictionary structure:\n---\n{}', hfp.datastructure2string())
        hfp.logging('')
        hfp.stoplogger()

    except:

        hfp.errout('Unexpected error')
