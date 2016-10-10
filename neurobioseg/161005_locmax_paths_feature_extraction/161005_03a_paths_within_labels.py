
from image_processing import ImageFileProcessing
from hdf5_processing import Hdf5Processing
import random
import vigra
import numpy as np
import os
import vigra.graphs as graphs

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
    gridgr = graphs.gridGraph(ifp.shape('curlabel'))

    ifp.logging('gridgr.shape = {}'.format(gridgr.shape))

    indicator = indicator.astype(np.float32)
    gridgr_edgeind = graphs.edgeFeaturesFromImage(gridgr, indicator)
    instance = graphs.ShortestPathPathDijkstra(gridgr)

    # Get two local maxima
    indices = np.where(ifp.get_image('curlocmax') == 1)
    coords = zip(indices[0], indices[1], indices[2])
    ifp.logging('Local maxima coordinates: {}'.format(coords))

    ifp.set_data_dict({'pathsim': np.zeros(ifp.get_image('curlocmax').shape)}, append=True)

    ifp.logging('len(coords) = {}'.format(len(coords)))
    paths = []
    for i in xrange(0, len(coords)-1):

        for j in xrange(i+1, len(coords)):

            ifp.logging('---')
            ifp.logging('i = {0}; j = {1}'.format(i, j))

            source = coords[i]
            target = coords[j]

            targetNode = gridgr.coordinateToNode(target)
            sourceNode = gridgr.coordinateToNode(source)

            ifp.logging('Source = {}'.format(source))
            ifp.logging('Target = {}'.format(target))

            instance.run(gridgr_edgeind, sourceNode, target=targetNode)
            path = instance.path(pathType='coordinates')
            paths.append(path)

            # for coords in path:
            #     # ifp.logging('coords = {}'.format(coords))
            #     pass

            pathindices = np.swapaxes(path, 0, 1)
            ifp.get_image('pathsim')[pathindices[0], pathindices[1], pathindices[2]] = 1

    # ifp.concatenate('disttransf', 'paths', target='paths_over_dist')
    ifp.astype(np.uint8, ('pathsim', 'curlocmax'))
    # ifp.anytask(vigra.filters.multiBinaryDilation, ('paths', 'locmax'), 3)
    ifp.swapaxes(0, 2, ids=('pathsim', 'curlocmax', 'curdisttransf'))
    ifp.anytask(vigra.filters.discDilation, 2, ids=('pathsim', 'curlocmax'))
    ifp.set_data_dict({'paths_over_dist': np.array([ifp.get_image('pathsim'), ifp.get_image('curlocmax'), ifp.get_image('curdisttransf')])}, append=True)

    return paths


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'

    ifp = ImageFileProcessing(
        yaml=yamlfile,
        yamlspec={'image_path': 'intermedfolder', 'image_file': 'locmaxfile', 'image_names': ('locmaxnames', 0, 2)},
        asdict=True,
        keys=('disttransf', 'locmax')
    )
    params = ifp.get_params()
    thisparams = params['paths_within_labels']
    ifp.addfromfile(params['intermedfolder']+params['largeobjfile'], image_names=params['largeobjname'], ids='largeobj')

    ifp.startlogger(filename=params['intermedfolder'] + 'paths_within_labels.log', type='a')

    ifp.code2log(__file__)
    ifp.logging('')

    ifp.logging('yamlfile = {}', yamlfile)
    ifp.logging('ifp.get_data().keys() = {}', ifp.get_data().keys())
    ifp.logging('ifp.shape() = {}', ifp.shape())
    ifp.logging('{}', ifp.amax())

    hfp = Hdf5Processing()
    c = 0
    for lblo in ifp.label_bounds_iterator('largeobj', 'curlabel', ids=('locmax', 'disttransf'), targetids=('curlocmax', 'curdisttransf'),
                                          maskvalue=0, value=0, background=0):

        ifp.logging('------------\nCurrent label {} in iteration {}', lblo['label'], c)
        ifp.logging('Bounding box = {}', lblo['bounds'])

        if ifp.amax('curlocmax') == 1:
            ps = find_shortest_path(ifp, thisparams['penaltypower'])
            ifp.logging('Number of paths found: {}', len(ps))
            if ps:
                hfp.setdata({lblo['label']: ps}, append=True)
                ifp.write(filename='paths_over_dist_true_{}.h5'.format(lblo['label']), ids='paths_over_dist')

        else:
            ifp.logging('No local maxima found for this label.')

        c += 1
        # if c == 5:
        #     break

    hfp.write(filepath=params['intermedfolder']+params['pathstruefile'])

    ifp.logging('')
    ifp.stoplogger()