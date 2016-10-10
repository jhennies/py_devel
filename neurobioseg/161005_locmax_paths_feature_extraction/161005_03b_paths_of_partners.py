
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

__author__ = 'jhennies'


def find_shortest_path(ifp, penaltypower):

    # Modify distancetransform
    #
    # a) Invert: the lowest values (i.e. the lowest penalty for the shortest path detection) should be at the center of
    #    the current process
    ifp.invert_image(ids='curdisttransf')
    #
    # b) Set all values outside the process to infinity
    ifp.filter_values(ifp.amax('curdisttransf'), type='eq', setto=np.inf, ids='curdisttransf', targetids='curdisttransf_inf')
    #
    # c) Increase the value difference between pixels near the boundaries and pixels central within the processes
    #    This increases the likelihood of the paths to follow the center of processes, thus avoiding short-cuts
    ifp.power(penaltypower, ids='curdisttransf_inf')

    indicator = ifp.get_image('curdisttransf_inf')
    gridgr = graphs.gridGraph(ifp.shape('curlabelpair'))

    ifp.logging('gridgr.shape = {}'.format(gridgr.shape))

    indicator = indicator.astype(np.float32)
    gridgr_edgeind = graphs.edgeFeaturesFromImage(gridgr, indicator)
    instance = graphs.ShortestPathPathDijkstra(gridgr)

    # Get local maxima
    indices1 = np.where(ifp.get_image('curlocmax_unml1') == 1)
    coords1 = zip(indices1[0], indices1[1], indices1[2])
    ifp.logging('Local maxima coordinates of label1: {}', coords1)

    indices2 = np.where(ifp.get_image('curlocmax_unml2') == 1)
    coords2 = zip(indices2[0], indices2[1], indices2[2])
    ifp.logging('Local maxima coordinates of label2: {}', coords2)

    ifp.set_data_dict({'pathsim': np.zeros(ifp.get_image('curlocmax').shape)}, append=True)

    ifp.logging('len(coords1) = {}', len(coords1))
    ifp.logging('len(coords2) = {}', len(coords2))

    paths = []

    for i in xrange(0, len(coords1)):

        for j in xrange(0, len(coords2)):

            ifp.logging('---')
            ifp.logging('i = {}; j = {}', i, j)

            source = coords1[i]
            target = coords2[j]

            target_node = gridgr.coordinateToNode(target)
            source_node = gridgr.coordinateToNode(source)

            ifp.logging('Source = {}'.format(source))
            ifp.logging('Target = {}'.format(target))

            instance.run(gridgr_edgeind, source_node, target=target_node)
            path = instance.path(pathType='coordinates')
            paths.append(path)

            # for coords in path:
            #     # ifp.logging('coords = {}'.format(coords))
            #     pass

            pathindices = np.swapaxes(path, 0, 1)
            ifp.get_image('pathsim')[pathindices[0], pathindices[1], pathindices[2]] = 1

    # ifp.concatenate('disttransf', 'paths', target='paths_over_dist')
    ifp.astype(np.uint8, ('pathsim'))
    # ifp.anytask(vigra.filters.multiBinaryDilation, ('paths', 'locmax'), 3)
    ifp.swapaxes(0, 2, ids=('pathsim', 'curlabelpair', 'curdisttransf'))
    ifp.anytask(vigra.filters.discDilation, 2, ids='pathsim')
    ifp.logging('ifp.shape = {}', ifp.shape(ids=('pathsim', 'curlabelpair', 'curdisttransf')))
    ifp.set_data_dict({'paths_over_dist': np.array([ifp.get_image('pathsim'), ifp.get_image('curlabelpair'), ifp.get_image('curdisttransf')])}, append=True)

    return paths


if __name__ == '__main__':

    try:

        yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'

        ifp = ImageFileProcessing(
            yaml=yamlfile,
            yamlspec={'image_path': 'intermedfolder', 'image_file': 'locmaxfile', 'image_names': ('locmaxnames', 1, 3)},
            asdict=True,
            keys=('disttransf', 'locmax')
        )
        params = ifp.get_params()
        thisparams = params['paths_of_partners']
        ifp.addfromfile(params['intermedfolder']+params['largeobjmfile'], image_names=params['largeobjmnames'],
                        ids=['largeobjm', 'mergeids_small', 'mergeids_random', 'mergeids_all'])
        ifp.addfromfile(params['intermedfolder']+params['largeobjfile'], image_names=params['largeobjname'], ids='largeobj')

        ifp.startlogger(filename=params['intermedfolder'] + 'paths_of_partners.log', type='a')

        ifp.code2log(__file__)
        ifp.logging('')

        ifp.logging('yamlfile = {}', yamlfile)
        ifp.logging('ifp.get_data().keys() = {}', ifp.get_data().keys())
        ifp.logging('ifp.shape() = {}', ifp.shape())
        ifp.logging('ifp.amax() = {}', ifp.amax())

        hfp = Hdf5Processing()
        c = 0
        # These are the labels which were merged with a respective partner
        # labellist = ifp.get_image('mergeids_all')[:, 0]
        labellist = ifp.get_image('mergeids_all')
        # labellist = [9988]
        ifp.logging('labellist = {}', labellist)
        # for lblo in ifp.label_bounds_iterator('largeobjm', 'curlabel',
        #                                       ids=('locmax', 'disttransf', 'largeobj'),
        #                                       targetids=('curlocmax', 'curdisttransf', 'curlargeobj'),
        #                                       maskvalue=0, value=0, background=0, labellist=labellist,
        #                                       forcecontinue=True):

        for lblp in ifp.labelpair_bounds_iterator('largeobj', 'curlabelpair',
                                                 ids=('locmax', 'disttransf'),
                                                 targetids=('curlocmax', 'curdisttransf'),
                                                 maskvalue=0, value=0,
                                                 labellist=labellist, forcecontinue=False):

            # lblo['unml1'] = ifp.get_image('mergeids_all')[c, 0]
            # lblo['unml2'] = ifp.get_image('mergeids_all')[c, 1]
            # lblo['unml1'] = 9988
            # lblo['unml2'] = 4077
            # #
            # # # ifp.deepcopy_entry('curlargeobj', 'curmergedlabels')
            # # mergeobj = getlabel((lblo['unml2'], lblo['unml1']), ifp.get_image('largeobj'))
            # # ifp.get_image('curmergedlabels')[mergeobj > 0] = 1
            #
            ifp.logging('------------\nCurrent labelpair {} in iteration {}', lblp['labels'], c)
            ifp.logging('Bounding box = {}', lblp['bounds'])
            # ifp.logging('Current unmerged labels: {} and {}', lblp['unml1'], lblp['unml2'])

            # Within the iterator the local maxima within both merged objects are available
            # Now get the local maxima of both objects individually (the 'UNMerged Labels' = 'unml')
            ifp.getlabel(lblp['labels'][0], ids='curlabelpair', targetids='unml1')
            ifp.getlabel(lblp['labels'][1], ids='curlabelpair', targetids='unml2')
            ifp.mask_image(maskvalue=0, value=0,
               ids=('curlocmax', 'curlocmax'),
               ids2=('unml1', 'unml2'),
               targetids=('curlocmax_unml1', 'curlocmax_unml2')
            )

            # Find the shortest paths between both labels
            ifp.logging('ifp.amax = {}', ifp.amax())
            if ifp.amax('curlocmax_unml1')  == 1 and ifp.amax('curlocmax_unml2') == 1:
                ps = find_shortest_path(ifp, thisparams['penaltypower'])

                ifp.logging('Number of paths found: {}', len(ps))
                if ps:
                    hfp.setdata({'{}_{}'.format(lblp['labels'][0], lblp['labels'][1]): ps}, append=True)
                    ifp.write(filename='paths_over_dist_false_{}_{}.h5'.format(lblp['labels'][0], lblp['labels'][1]), ids='paths_over_dist')

            else:
                ifp.logging('No local maxima found for at least one partner.')

            c += 1
            # if c == 5:
            #     break

        hfp.write(filepath=params['intermedfolder']+params['pathsfalsefile'])

        ifp.logging('')
        ifp.stoplogger()

    except:

        ifp.errout('Unexpected error', traceback)

    try:
        hfp.write(filepath=params['intermedfolder'] + params['pathsfalsefile'])
    except:
        pass
