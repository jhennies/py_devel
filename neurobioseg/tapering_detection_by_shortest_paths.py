
from image_processing import ImageProcessing, ImageFileProcessing
import numpy as np
import vigra
import sys
from vigra import graphs

__author__ = 'jhennies'

# General TODOs
# Done: Find local maxima, maybe after gaussian smoothing (or other smoothing)
# TODO: Determine shortest paths pairwise between maxima
# TODO: Extract features along path (e.g., distance transform values)
# TODO: What do these features look like along correctly segmented backbones and what they look like at false merges
# TODO: For the above: Implement randomly merged objects within the ground truth

# Specific TODOs
# Done: Re-implement image_processing: ImageFileProcessing should inherit ImageProcessing!
# Done: Implement generator to iterate over each label in the image data
# Done: Implement cropping to ImageFileProcessing and ImageProcessing


def gaussian_smoothing(image, sigma, roi=None):
    return vigra.filters.gaussianSmoothing(image, sigma)


def find_local_maxima(ifp):

    ifp.deepcopy_entry('currentlabel', 'locmax')

    # Distance transform
    ifp.invert_image(('locmax',))
    ifp.distance_transform(pixel_pitch=anisotropy, ids=('locmax',))

    # Smoothing
    ifp.anytask(gaussian_smoothing, ('locmax',), anisotropy/10)

    # Local maxima
    ifp.anytask(vigra.analysis.extendedLocalMaxima3D, ('locmax',), neighborhood=26)


def find_shortest_path(ifp):

    gridgr = graphs.gridGraph(ifp.shape('labels'))
    ifp.logging('ifp.shape(tuple) = {}', ifp.shape(('labels',)))
    ifp.logging('ifp.shape(str) = {}', ifp.shape('labels'))
    pass


if __name__ == '__main__':

    # ifp = ImageFileProcessing.empty()

    crop = True
    anisotropy = np.array([10, 1, 1])

    folder = '/media/julian/Daten/neuraldata/cremi_2016/'
    file = 'cremi.splA.raw_neurons.crop.h5'
    names = ('neuron_ids',)
    keys = ('labels',)

    ifp = ImageFileProcessing(
        folder,
        file, asdict=True,
        image_names=names,
        keys=keys)

    ifp.startlogger(filename=None, type='w')

    # Cropping
    if crop:
        ifp.crop([10, 200, 200], [110, 712, 712])
        # ifp.write()


    ifp.logging('ifp.get_image = {}', ifp.get_image('labels')[0, 0, 0])
    ifp.logging('ifp.amax = {}\n', ifp.amax(('labels',)))

    # for lbl in ifp.label_iterator('labels'):
    #     ifp.logging('label_iterator: {}', lbl)

    c = 0
    for lbl in ifp.label_image_iterator('labels', 'currentlabel'):
        ifp.logging('Current label = {}', lbl)

        find_local_maxima(ifp)
        find_shortest_path(ifp)

        ifp.write(filename='test/test_{}.h5'.format(c))

        c += 1
        if c == 2:
            break

    ifp.logging('')

    ifp.stoplogger()

    # for i in ifp.label_iterator():
    #     print i