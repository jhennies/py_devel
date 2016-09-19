
from image_processing import ImageProcessing, ImageFileProcessing

__author__ = 'jhennies'

# General TODOs
# TODO: Find local maxima, maybe after gaussian smoothing (or other smoothing)
# TODO: Determine shortest paths pairwise between maxima
# TODO: Extract features along path (e.g., distance transform values)
# TODO: What do these features look like along correctly segmented backbones and what they look like at false merges
# TODO: For the above: Implement randomly merged objects within the ground truth

# Specific TODOs
# TODO: Re-implement image_processing: ImageFileProcessing should inherit ImageProcessing!
# TODO: Implement generator to iterate over each label in the image data

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

    print ifp.get_image('labels')[0,0,0]

    def plus1(image):

        return image + 1

    ifp.anytask(plus1, '', ('labels',))

    print ifp.get_image('labels')[0,0,0]

    print ifp.amax(('labels',))


    # for i in ifp.label_iterator():
    #     print i