
from image_processing import ImageProcessing, ImageFileProcessing
import numpy as np

__author__ = 'jhennies'

# General TODOs
# TODO: Find local maxima, maybe after gaussian smoothing (or other smoothing)
# TODO: Determine shortest paths pairwise between maxima
# TODO: Extract features along path (e.g., distance transform values)
# TODO: What do these features look like along correctly segmented backbones and what they look like at false merges
# TODO: For the above: Implement randomly merged objects within the ground truth

# Specific TODOs
# Done: Re-implement image_processing: ImageFileProcessing should inherit ImageProcessing!
# Done: Implement generator to iterate over each label in the image data

if __name__ == '__main__':

    # ifp = ImageFileProcessing.empty()

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

    ifp.logging('ifp.get_image = {}', ifp.get_image('labels')[0, 0, 0])
    ifp.logging('ifp.amax = {}\n', ifp.amax(('labels',)))

    # for lbl in ifp.label_iterator('labels'):
    #     ifp.logging('label_iterator: {}', lbl)

    c = 0
    for lbl in ifp.label_image_iterator('labels', 'currentlabel'):
        ifp.logging('Current label = {}', lbl)
        # ifp.write(filename='test_{}.h5'.format(c))


        c += 1
        if c == 20:
            break

    ifp.logging('')

    ifp.stoplogger()

    # for i in ifp.label_iterator():
    #     print i