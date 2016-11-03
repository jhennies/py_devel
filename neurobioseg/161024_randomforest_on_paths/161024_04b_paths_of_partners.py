
from hdf5_image_processing import Hdf5ImageProcessingLib as IPL
from image_processing import ImageFileProcessing
from processing_lib import getlabel
from hdf5_processing import Hdf5Processing
import random
import vigra
import numpy as np
import os
import vigra.graphs as graphs
import sys
import traceback
import inspect
from shutil import copy
import processing_lib as lib


__author__ = 'jhennies'


# def find_shortest_path(ifp, penaltypower, bounds):
#
#     # Modify distancetransform
#     #
#     # a) Invert: the lowest values (i.e. the lowest penalty for the shortest path detection) should be at the center of
#     #    the current process
#     ifp.invert_image(ids='curdisttransf')
#     #
#     # b) Set all values outside the process to infinity
#     ifp.filter_values(ifp.amax('curdisttransf'), type='eq', setto=np.inf, ids='curdisttransf', targetids='curdisttransf_inf')
#     #
#     # c) Increase the value difference between pixels near the boundaries and pixels central within the processes
#     #    This increases the likelihood of the paths to follow the center of processes, thus avoiding short-cuts
#     ifp.power(penaltypower, ids='curdisttransf_inf')
#
#     indicator = ifp.get_image('curdisttransf_inf')
#     gridgr = graphs.gridGraph(ifp.shape('curlabelpair'))
#
#     ifp.logging('gridgr.shape = {}'.format(gridgr.shape))
#
#     indicator = indicator.astype(np.float32)
#     gridgr_edgeind = graphs.edgeFeaturesFromImage(gridgr, indicator)
#     instance = graphs.ShortestPathPathDijkstra(gridgr)
#
#     # Get local maxima
#     indices1 = np.where(ifp.get_image('curlocmax_unml1') == 1)
#     coords1 = zip(indices1[0], indices1[1], indices1[2])
#     ifp.logging('Local maxima coordinates of label1: {}', coords1)
#
#     indices2 = np.where(ifp.get_image('curlocmax_unml2') == 1)
#     coords2 = zip(indices2[0], indices2[1], indices2[2])
#     ifp.logging('Local maxima coordinates of label2: {}', coords2)
#
#     ifp.set_data_dict({'pathsim': np.zeros(ifp.get_image('curlocmax').shape)}, append=True)
#
#     ifp.logging('len(coords1) = {}', len(coords1))
#     ifp.logging('len(coords2) = {}', len(coords2))
#
#     paths = []
#
#     for i in xrange(0, len(coords1)):
#
#         for j in xrange(0, len(coords2)):
#
#             ifp.logging('---')
#             ifp.logging('i = {}; j = {}', i, j)
#
#             source = coords1[i]
#             target = coords2[j]
#
#             target_node = gridgr.coordinateToNode(target)
#             source_node = gridgr.coordinateToNode(source)
#
#             ifp.logging('Source = {}'.format(source))
#             ifp.logging('Target = {}'.format(target))
#
#             instance.run(gridgr_edgeind, source_node, target=target_node)
#             path = instance.path(pathType='coordinates')
#             # Do not forget to correct for the offset caused by cropping!
#             paths.append(path + [bounds[0][0], bounds[1][0], bounds[2][0]])
#
#             # for coords in path:
#             #     # ifp.logging('coords = {}'.format(coords))
#             #     pass
#
#             pathindices = np.swapaxes(path, 0, 1)
#             ifp.get_image('pathsim')[pathindices[0], pathindices[1], pathindices[2]] = 1
#
#     # ifp.concatenate('disttransf', 'paths', target='paths_over_dist')
#     ifp.astype(np.uint8, ('pathsim'))
#     # ifp.anytask(vigra.filters.multiBinaryDilation, ('paths', 'locmax'), 3)
#     ifp.swapaxes(0, 2, ids=('pathsim', 'curlabelpair', 'curdisttransf'))
#     ifp.anytask(vigra.filters.discDilation, 2, ids='pathsim')
#     ifp.logging('ifp.shape = {}', ifp.shape(ids=('pathsim', 'curlabelpair', 'curdisttransf')))
#     ifp.set_data_dict({'paths_over_dist': np.array([ifp.get_image('pathsim'), ifp.get_image('curlabelpair'), ifp.get_image('curdisttransf')])}, append=True)
#
#     return paths

def find_shortest_path(hfp, penaltypower, bounds, disttransf, locmax,
                       labels, labelgroup):

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

    # The situation:
    # We have multiple objects (n > 1) of unknown number.
    # We want to extract the local maxima within each object individually and create a
    #   list of all possible partners (pairs)
    # Each partner of a pair has to be located within a different object (label area)
    #
    # Approach 1:
    #   For each locmax iterate over all other locmaxs and write pairs, which satisfy the
    #   desired condition, to the pairs list
    #
    # Approach 2:
    #   For each label object iterate over all the others and append all possible locmax
    #   pairs to the pairs list
    #   Probably faster than approach 1 when implemented correctly? Someone should test that...

    # Approach 2 in its implemented form
    pairs = []
    for i in xrange(0, len(labelgroup)-1):
        indices_i = np.where((labels == labelgroup[i]) & (locmax > 0))
        indices_i = zip(indices_i[0], indices_i[1], indices_i[2])
        if indices_i:
            for j in xrange(i+1, len(labelgroup)):
                indices_j = np.where((labels == labelgroup[j]) & (locmax > 0))
                indices_j = zip(indices_j[0], indices_j[1], indices_j[2])
                if indices_j:
                    hfp.logging('Ind_i = {}\nInd_j = {}', indices_i, indices_j)
                    # Now, lets do some magic!
                    pairs = pairs + zip(indices_i * len(indices_j), sorted(indices_j * len(indices_i)))

    paths, pathim = lib.shortest_paths(disttransf, pairs, bounds=bounds, hfp=hfp)

    return paths, pathim


def paths_of_partners(hfp, lblsmkey, lblskey, changehashkey, locmaxkeys, disttransfkey, ignore):

    params = hfp.get_params()
    thisparams = params['paths_of_partners']

    if type(locmaxkeys) is str:
        locmaxkeys = (locmaxkeys,)

    # This results in the format:
    # more_keys = (locmaxkeys[0], ..., locmaxkeys[n], disttransfkey, lblskey)
    more_keys = locmaxkeys + (disttransfkey,) + (lblskey,)

    paths = IPL()
    for k in locmaxkeys:
        paths['pathsim', k] = np.zeros(hfp[lblsmkey].shape)

    # Determine labellist from change_hash (keys are strings due to h5 saving)
    labellist = hfp[changehashkey].keys()
    labellist = [int(x) for x in labellist]

    for lbl, lblim, more_ims, bounds in hfp.label_image_bounds_iterator(
        key=lblsmkey, background=0, more_keys=more_keys,
        maskvalue=0, value=0, labellist=labellist
    ):
        if lbl in ignore:
            continue
        # The format of more_ims is:
        # more_ims = {locmaxkeys[0]: locmax_1,
        #                ...
        #             locmaxkeys[n]: locmax_n,
        #             disttransfkey: disttransf,
        #             lblskey: lables}

        hfp.logging('======================\nWorking on label = {}', lbl)
        # hfp.logging('more_ims structure:\n---\n{}', more_ims.datastructure2string())
        hfp.logging('bounds = {}', bounds)

        for i in xrange(0, len(locmaxkeys)):

            hfp.logging('locmaxkey = {}', locmaxkeys[i])
            if np.amax(more_ims[locmaxkeys[i]]) > 0:
                ps, pathsim = find_shortest_path(hfp, thisparams['penaltypower'], bounds,
                                                 more_ims[disttransfkey],
                                                 more_ims[locmaxkeys[i]],
                                                 more_ims[lblskey],
                                                 hfp[changehashkey][str(lbl)])

                # Only store the path if the path-calculation successfully determined a path
                # Otherwise an empty list would be stored
                if ps:
                    hfp.logging('Number of paths found: {}', len(ps))

                    paths[lblsmkey, locmaxkeys[i], 'path', lbl] = ps
                    paths[lblsmkey, locmaxkeys[i], 'pathsim'] = pathsim

                    paths['pathsim', locmaxkeys[i]][bounds][pathsim > 0] = pathsim[pathsim > 0]

    for k in locmaxkeys:
        paths['overlay', k] = np.array([paths['pathsim', k],
                                     hfp[lblskey].astype(np.float32) / np.amax(hfp[lblskey]),
                                     vigra.filters.multiBinaryDilation(hfp[k].astype(np.uint8), 5)])

    return paths


# if False:
#
#     try:
#
#         yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'
#
#         ifp = ImageFileProcessing(
#             yaml=yamlfile,
#             yamlspec={'image_path': 'intermedfolder', 'image_file': 'locmaxfile', 'image_names': ('locmaxnames', 1, 3)},
#             asdict=True,
#             keys=('disttransf', 'locmax')
#         )
#         params = ifp.get_params()
#         thisparams = params['paths_of_partners']
#         ifp.addfromfile(params['intermedfolder']+params['largeobjmfile'], image_names=params['largeobjmnames'],
#                         ids=['largeobjm', 'mergeids_small', 'mergeids_random', 'mergeids_all'])
#         ifp.addfromfile(params['intermedfolder']+params['largeobjfile'], image_names=params['largeobjname'], ids='largeobj')
#
#         ifp.startlogger(filename=params['intermedfolder'] + 'paths_of_partners.log', type='a')
#
#         # ifp.code2log(__file__)
#         ifp.code2log(inspect.stack()[0][1])
#         ifp.logging('')
#
#         ifp.logging('yamlfile = {}', yamlfile)
#         ifp.logging('ifp.get_data().keys() = {}', ifp.get_data().keys())
#         ifp.logging('ifp.shape() = {}', ifp.shape())
#         ifp.logging('ifp.amax() = {}', ifp.amax())
#
#         hfp = Hdf5Processing()
#         c = 0
#         # These are the labels which were merged with a respective partner
#         # labellist = ifp.get_image('mergeids_all')[:, 0]
#         labellist = ifp.get_image('mergeids_all')
#         # labellist = [9988]
#         ifp.logging('labellist = {}', labellist)
#         # for lblo in ifp.label_bounds_iterator('largeobjm', 'curlabel',
#         #                                       ids=('locmax', 'disttransf', 'largeobj'),
#         #                                       targetids=('curlocmax', 'curdisttransf', 'curlargeobj'),
#         #                                       maskvalue=0, value=0, background=0, labellist=labellist,
#         #                                       forcecontinue=True):
#
#         for lblp in ifp.labelpair_bounds_iterator('largeobj', 'curlabelpair',
#                                                  ids=('locmax', 'disttransf'),
#                                                  targetids=('curlocmax', 'curdisttransf'),
#                                                  maskvalue=0, value=0,
#                                                  labellist=labellist, forcecontinue=False):
#
#             # lblo['unml1'] = ifp.get_image('mergeids_all')[c, 0]
#             # lblo['unml2'] = ifp.get_image('mergeids_all')[c, 1]
#             # lblo['unml1'] = 9988
#             # lblo['unml2'] = 4077
#             # #
#             # # # ifp.deepcopy_entry('curlargeobj', 'curmergedlabels')
#             # # mergeobj = getlabel((lblo['unml2'], lblo['unml1']), ifp.get_image('largeobj'))
#             # # ifp.get_image('curmergedlabels')[mergeobj > 0] = 1
#             #
#             ifp.logging('------------\nCurrent labelpair {} in iteration {}', lblp['labels'], c)
#             ifp.logging('Bounding box = {}', lblp['bounds'])
#             # ifp.logging('Current unmerged labels: {} and {}', lblp['unml1'], lblp['unml2'])
#
#             # Within the iterator the local maxima within both merged objects are available
#             # Now get the local maxima of both objects individually (the 'UNMerged Labels' = 'unml')
#             ifp.getlabel(lblp['labels'][0], ids='curlabelpair', targetids='unml1')
#             ifp.getlabel(lblp['labels'][1], ids='curlabelpair', targetids='unml2')
#             ifp.mask_image(maskvalue=0, value=0,
#                ids=('curlocmax', 'curlocmax'),
#                ids2=('unml1', 'unml2'),
#                targetids=('curlocmax_unml1', 'curlocmax_unml2')
#             )
#
#             # Find the shortest paths between both labels
#             ifp.logging('ifp.amax = {}', ifp.amax())
#             if ifp.amax('curlocmax_unml1')  == 1 and ifp.amax('curlocmax_unml2') == 1:
#                 ps = find_shortest_path(ifp, thisparams['penaltypower'], lblp['bounds'])
#
#                 ifp.logging('Number of paths found: {}', len(ps))
#                 if ps:
#                     hfp.setdata({'{}_{}'.format(lblp['labels'][0], lblp['labels'][1]): ps}, append=True)
#                     ifp.write(filename='paths_over_dist_false_{}_{}.h5'.format(lblp['labels'][0], lblp['labels'][1]), ids='paths_over_dist')
#
#             else:
#                 ifp.logging('No local maxima found for at least one partner.')
#
#             c += 1
#             # if c == 5:
#             #     break
#
#         hfp.write(filepath=params['intermedfolder']+params['pathsfalsefile'])
#
#         ifp.logging('')
#         ifp.stoplogger()
#
#     except:
#
#         ifp.errout('Unexpected error', traceback)
#
#     try:
#         hfp.write(filepath=params['intermedfolder'] + params['pathsfalsefile'])
#     except:
#         pass


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'

    hfp = IPL(
        yaml=yamlfile,
        yamlspec={'path': 'intermedfolder', 'filename': 'locmaxborderfile', 'skeys': {'locmaxbordernames': (1, 3)}},
        tkeys=('border_locmax_m', 'disttransfm'),
        castkey=None
    )
    params = hfp.get_params()
    thisparams = params['paths_of_partners']
    hfp.startlogger(filename=params['resultfolder'] + 'paths_of_partners.log', type='w')
    hfp.data_from_file(params['intermedfolder'] + params['largeobjfile'],
                       skeys=params['largeobjname'],
                       tkeys='largeobj')
    hfp.data_from_file(params['intermedfolder'] + params['largeobjmfile'],
                       skeys=(params['largeobjmnames'][0], params['largeobjmnames'][4]),
                       tkeys=('largeobjm', 'change_hash'))
    hfp.data_from_file(params['intermedfolder'] + params['locmaxfile'],
                       skeys=params['locmaxnames'][0],
                       tkeys='locmaxm')

    try:

        # Copy the script file and the parameters to the scriptsfolder
        copy(inspect.stack()[0][1], params['scriptsfolder'])
        copy(yamlfile, params['scriptsfolder'] + 'paths_of_partners.parameters.yml')
        # Write script and parameters to the logfile
        hfp.code2log(inspect.stack()[0][1])
        hfp.logging('')
        hfp.yaml2log()
        hfp.logging('')

        hfp.logging('\nhfp datastructure: \n\n{}', hfp.datastructure2string(maxdepth=1))

        paths = paths_of_partners(hfp, 'largeobjm', 'largeobj',
                                  'change_hash', ('border_locmax_m', 'locmaxm'),
                                  'disttransfm', thisparams['ignore'])

        paths.write(filepath=params['intermedfolder'] + params['pathsfalsefile'])

        hfp.logging('\nFinal dictionary structure:\n---\n{}', hfp.datastructure2string())
        hfp.logging('')
        hfp.stoplogger()

    except:

        hfp.errout('Unexpected error')
