
from image_processing import ImageProcessing, ImageFileProcessing
import numpy as np
import vigra
import sys
from vigra import graphs
import scipy as sp
import copy

__author__ = 'jhennies'

# General TODOs
# Done: Find local maxima, maybe after gaussian smoothing (or other smoothing)
# Done: Determine shortest paths pairwise between maxima
# Done: Optimize paths such that they follow the center of processes
# TODO: Extract features along path (e.g., distance transform values)
# TODO: What do these features look like along correctly segmented backbones and what they look like at false merges
# TODO: For the above: Implement randomly merged objects within the ground truth
# TODO: Random Forest to classify paths between two maxima either as correct or false merge

# Specific TODOs
# Done: Re-implement image_processing: ImageFileProcessing should inherit ImageProcessing!
# Done: Implement generator to iterate over each label in the image data
# Done: Implement cropping to ImageFileProcessing and ImageProcessing
# Done: Overlay images -> or just save as multiple channels
# Done: Optimize code: anytask function should be able to compute results depending on multiple images
# TODO: Rewrite to load distance transform from hard disk


def gaussian_smoothing(image, sigma, roi=None):
    return vigra.filters.gaussianSmoothing(image, sigma)


def find_local_maxima(ifp):

    # ifp.deepcopy_entry('currentlabel', 'disttransf')

    # # Distance transform
    # ifp.invert_image(ids='currentlabel', targetids='disttransf')
    # ifp.distance_transform(pixel_pitch=anisotropy, ids='disttransf')

    # Smoothing
    # ifp.deepcopy_entry('disttransf', 'smoothed')
    try:
        ifp.anytask(gaussian_smoothing, 20/anisotropy, ids='curdisttransf', targetids='smoothed')

        # Local maxima
        # ifp.deepcopy_entry('smoothed', 'locmax')
        ifp.astype(np.float32, ids='smoothed')
        ifp.anytask(vigra.analysis.extendedLocalMaxima3D, neighborhood=26, ids='smoothed', targetids='locmax')
        # ifp.anytask(filters.maximum_filter, 'locmax', 5)

    except:

        return False

    return True


def find_shortest_path(ifp):

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
    ifp.power(10, ids='curdisttransf_inf')

    indicator = ifp.get_image('curdisttransf_inf')
    gridgr = graphs.gridGraph(ifp.shape('curlabel'))

    ifp.logging('gridgr.shape = {}'.format(gridgr.shape))

    indicator = indicator.astype(np.float32)
    gridgr_edgeind = graphs.edgeFeaturesFromImage(gridgr, indicator)
    instance = graphs.ShortestPathPathDijkstra(gridgr)

    # Get two local maxima
    indices = np.where(ifp.get_image('locmax') == 1)
    coords = zip(indices[0], indices[1], indices[2])
    ifp.logging('Local maxima coordinates: {}'.format(coords))

    # ifp.deepcopy_entry('locmax', 'paths')
    ifp.set_data_dict({'paths': np.zeros(ifp.get_image('locmax').shape)}, append=True)

    ifp.logging('len(coords) = {}'.format(len(coords)))
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

            # for coords in path:
            #     # ifp.logging('coords = {}'.format(coords))
            #     pass

            pathindices = np.swapaxes(path, 0, 1)
            ifp.get_image('paths')[pathindices[0], pathindices[1], pathindices[2]] = 1

    # ifp.concatenate('disttransf', 'paths', target='paths_over_dist')
    ifp.astype(np.uint8, ('paths', 'locmax'))
    # ifp.anytask(vigra.filters.multiBinaryDilation, ('paths', 'locmax'), 3)
    ifp.swapaxes(0, 2, ids=('paths', 'locmax', 'curdisttransf'))
    ifp.anytask(vigra.filters.discDilation, 2, ids=('paths', 'locmax'))
    ifp.set_data_dict({'paths_over_dist': np.array([ifp.get_image('paths'), ifp.get_image('locmax'), ifp.get_image('curdisttransf')])}, append=True)


if __name__ == '__main__':

    # ifp = ImageFileProcessing.empty()


    crop = True
    anisotropy = np.array([10, 1, 1])

    folder = '/media/julian/Daten/neuraldata/cremi_2016/'
    file = 'cremi.splA.raw_neurons.crop.crop_10-200-200_110-712-712.disttransf.h5'
    names = ('labels','disttransf')
    keys = ('labels','disttransf')
    resultfolder = 'develop/160927_determine_shortest_paths/160929_pthsovrdist_pow10/'
    if True:

        ifp = ImageFileProcessing(
        folder,
        file, asdict=True,
        image_names=names,
        keys=keys)

        ifp.startlogger(filename='/media/julian/Daten/neuraldata/cremi_2016/develop/160927_determine_shortest_paths/160929_pthsovrdist_pow10/160929.log', type='a')

        # # Cropping
        # if crop:
        #     ifp.crop([10, 200, 200], [110, 712, 712])

            # ifp.write()

        # ifp.logging('ifp.get_image = {}', ifp.get_image('labels')[0, 0, 0])
        # ifp.logging('ifp.amax = {}\n', ifp.amax('labels'))

        ifp.logging('keys() = {}', ifp.get_data().keys())

        # For multiple labels
        ifp.logging('Starting label iterator for {} labels', len(ifp.anytask_rtrn(np.unique, ids='labels')))

        c = 0
        for lblo in ifp.label_bounds_iterator('labels', 'curlabel', ids='disttransf', targetids='curdisttransf',
                                              maskvalue=0, value=0, minsize=[10, 100, 100]):

            ifp.logging('------------\nCurrent label {} in iteration {}', lblo['label'], c)
            ifp.logging('Bounding box = {}', lblo['bounds'])

            local_maxima_found = find_local_maxima(ifp)

            ifp.logging('Local maxima found: {}', local_maxima_found)

            if local_maxima_found:

                if ifp.amax('locmax') != 0:

                    find_shortest_path(ifp)

                    ifp.write(filename='{0}input_{1}.h5'.format(resultfolder, lblo['label']), ids=('curdisttransf', 'curlabel', 'smoothed'))
                    ifp.write(filename='{0}paths_over_dist_{1}.h5'.format(resultfolder, lblo['label']), ids=('paths_over_dist',))

                else:

                    ifp.logging('No maxima found!')

            else:

                ifp.logging('Maxima detection not successful!')


            c += 1
            # if c == 500:
            #     break

        # c = 0
        # for lbl in ifp.label_image_iterator('labels', 'currentlabel'):
        #
        #     ifp.logging('------------\nCurrent label {} in iteration {}', lbl, c)
        #
        #     find_local_maxima(ifp)
        #
        #     if ifp.amax('locmax') != 0:
        #         find_shortest_path(ifp)
        #         ifp.write(
        #             filename='develop/160927_determine_shortest_paths/160927_pthsovrdist_pow10_morelabels/after_shortest_path_{}.h5'.format(
        #                 lbl), ids=('disttransf', 'currentlabel', 'paths', 'locmax', 'smoothed', 'disttransf_inf'))
        #         ifp.write(
        #             filename='develop/160927_determine_shortest_paths/160927_pthsovrdist_pow10_morelabels/paths_over_dist_{}.h5'.format(
        #                 lbl), ids=('paths_over_dist',))
        #     else:
        #         ifp.logging('Non maxima found!')
        #
        #     c += 1
        #     # if c == 10:
        #     #     break

    else:
        lbl = 10230
        ifp = ImageFileProcessing(
            folder,
            'test/after_local_maxima_10230.h5',
            asdict=True
        )

    # if True:
    #     find_shortest_path(ifp)
    #     ifp.write(filename='test/after_shortest_path_{}.h5'.format(lbl), ids=('disttransf', 'currentlabel', 'paths', 'locmax', 'smoothed', 'disttransf_inf'))
    #     ifp.write(filename='test/paths_over_dist_{}.h5'.format(lbl), ids=('paths_over_dist',))
    # else:
    #     ifp = ImageFileProcessing(
    #         folder + 'test/',
    #         'after_shortest_path_{}.h5'.format(lbl),
    #         asdict=True
    #     )

    ifp.logging('')

    ifp.stoplogger()

    # for i in ifp.label_iterator():
    #     print i