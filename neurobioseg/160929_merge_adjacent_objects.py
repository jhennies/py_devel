
from image_processing import ImageFileProcessing
import random

__author__ = 'jhennies'

if __name__ == '__main__':

    folder = '/media/julian/Daten/neuraldata/cremi_2016/'
    file = 'cremi.splA.raw_neurons.crop.crop_10-200-200_110-712-712.h5'
    names = ('labels',)
    keys = ('labels',)
    resultfolder = 'develop/160927_determine_shortest_paths/160929_pthsovrdist_pow10/'

    ifp = ImageFileProcessing(
        folder,
        file, asdict=True,
        image_names=names,
        keys=keys)

    ifp.startlogger(filename=None, type='a')

    ifp.logging('keys = {}', ifp.get_data().keys())

    # TODO: Randomly select a label
    # TODO: Find adjacent objects
    # TODO: Randomly select an adjacent object for merging
    # TODO: Merge the two

    # TODO: Randomly select a label
    # Find all labels
    labels = ifp.unique(ids='labels')
    ifp.logging('labels = {}', labels)
    random.seed()
    ifp.logging('Choice: {}', random.choice(labels))
    ifp.logging('Choice: {}', random.choice(labels))
    ifp.logging('Choice: {}', random.choice(labels))
    ifp.logging('Choice: {}', random.choice(labels))
    ifp.logging('Choice: {}', random.choice(labels))
    ifp.logging('Choice: {}', random.choice(labels))
    ifp.logging('Choice: {}', random.choice(labels))

    ifp.logging('')
    ifp.stoplogger()