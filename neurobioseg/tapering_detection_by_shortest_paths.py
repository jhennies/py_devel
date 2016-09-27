
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

# Specific TODOs
# Done: Re-implement image_processing: ImageFileProcessing should inherit ImageProcessing!
# Done: Implement generator to iterate over each label in the image data
# Done: Implement cropping to ImageFileProcessing and ImageProcessing
# Done: Overlay images -> or just save as multiple channels
# Done: Optimize code: anytask function should be able to compute results depending on multiple images


def gaussian_smoothing(image, sigma, roi=None):
    return vigra.filters.gaussianSmoothing(image, sigma)


def find_local_maxima(ifp):

    # ifp.deepcopy_entry('currentlabel', 'disttransf')

    # Distance transform
    ifp.invert_image(ids='currentlabel', targetids='disttransf')
    ifp.distance_transform(pixel_pitch=anisotropy, ids='disttransf')

    # Smoothing
    # ifp.deepcopy_entry('disttransf', 'smoothed')
    ifp.anytask(gaussian_smoothing, 20/anisotropy, ids='disttransf', targetids='smoothed')

    # Local maxima
    # ifp.deepcopy_entry('smoothed', 'locmax')
    ifp.anytask(vigra.analysis.extendedLocalMaxima3D, neighborhood=26, ids='smoothed', targetids='locmax')
    # ifp.anytask(filters.maximum_filter, 'locmax', 5)


def find_shortest_path(ifp):

    ifp.invert_image(ids='disttransf')
    # ifp.deepcopy_entry('disttransf', 'disttransf_inf')
    # ifp.anytask(vigra.filters.gaussianGradientMagnitude, 'gradmag', [0.1, 1, 1])
    ifp.filter_values(ifp.amax('disttransf'), type='eq', setto=np.inf, ids='disttransf', targetids='disttransf_inf')
    # def mult(image, value):
    #     return image * value
    # def pow(image, value):
    #     return np.power(image, value)
    # ifp.anytask(pow, 10, ids='disttransf_inf')

    # Increase the value difference between pixels near the boundaries and pixels central within the processes
    # This increases the likelihood of the paths to follow the center of processes, thus avoiding short-cuts
    ifp.power(10, ids='disttransf_inf')

    # indicator = copy.deepcopy(ifp.get_image('gradmag'))
    # disttransf = ifp.get_image('disttransf')
    # indicator[disttransf == np.amax(disttransf)] = np.inf
    # ifp.set_data_dict({'gradmag_inf': indicator}, append=True)
    # indicator = ifp.get_image('gradmag_inf')

    # ifp.invert_image(ids='smoothed')

    indicator = ifp.get_image('disttransf_inf')
    gridgr = graphs.gridGraph(ifp.shape('labels'))

    ifp.logging('gridgr.shape = {}'.format(gridgr.shape))

    gridgr_edgeind = graphs.edgeFeaturesFromImage(gridgr, indicator)
    # print gridgr_edgeind.shape
    instance = graphs.ShortestPathPathDijkstra(gridgr)
    # print instance

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
    ifp.swapaxes(0, 2, ids=('paths', 'locmax', 'disttransf'))
    ifp.anytask(vigra.filters.discDilation, 2, ids=('paths', 'locmax'))
    ifp.set_data_dict({'paths_over_dist': np.array([ifp.get_image('paths'), ifp.get_image('locmax'), ifp.get_image('disttransf')])}, append=True)


if __name__ == '__main__':

    # ifp = ImageFileProcessing.empty()


    crop = True
    anisotropy = np.array([10, 1, 1])

    folder = '/media/julian/Daten/neuraldata/cremi_2016/'
    file = 'cremi.splA.raw_neurons.crop.h5'
    names = ('neuron_ids',)
    keys = ('labels',)

    if True:

        ifp = ImageFileProcessing(
        folder,
        file, asdict=True,
        image_names=names,
        keys=keys)

        ifp.startlogger(filename='/media/julian/Daten/neuraldata/cremi_2016/develop/160927_determine_shortest_paths/160927_pthsovrdist_pow10_morelabels/160927.log', type='w')

        # Cropping
        if crop:
            ifp.crop([10, 200, 200], [110, 712, 712])

            # ifp.write()

        ifp.logging('ifp.get_image = {}', ifp.get_image('labels')[0, 0, 0])
        ifp.logging('ifp.amax = {}\n', ifp.amax('labels'))

        # For multiple labels
        ifp.logging('Starting label iterator for {} labels', len(ifp.anytask_rtrn(np.unique, ids='labels')))
        c = 0
        for lbl in ifp.label_image_iterator('labels', 'currentlabel'):

            ifp.logging('------------\nCurrent label {} in iteration {}', lbl, c)

            find_local_maxima(ifp)

            if ifp.amax('locmax') != 0:
                find_shortest_path(ifp)
                ifp.write(
                    filename='develop/160927_determine_shortest_paths/160927_pthsovrdist_pow10_morelabels/after_shortest_path_{}.h5'.format(
                        lbl), ids=('disttransf', 'currentlabel', 'paths', 'locmax', 'smoothed', 'disttransf_inf'))
                ifp.write(
                    filename='develop/160927_determine_shortest_paths/160927_pthsovrdist_pow10_morelabels/paths_over_dist_{}.h5'.format(
                        lbl), ids=('paths_over_dist',))
            else:
                ifp.logging('Non maxima found!')


            c += 1
            # if c == 10:
            #     break

        # # DEBUG: For one label only
        # # it = ifp.label_image_iterator('labels', 'currentlabel')
        # # lbl = it.next()
        # lbl = 10230
        # # ifp.deepcopy_entry('labels', 'currentlabel')
        # # ifp.getlabel(10230, ('currentlabel',))
        # ifp.getlabel(10230, ids='labels', targetids='currentlabel')
        # ifp.logging('Current label = {}', lbl)
        #
        # find_local_maxima(ifp)
        # ifp.write(filename='test/after_local_maxima_{}.h5'.format(lbl))

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