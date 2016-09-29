
from image_processing import ImageFileProcessing
import vigra
import numpy as np

__author__ = 'jhennies'


def gaussian_smoothing(image, sigma, roi=None):
    return vigra.filters.gaussianSmoothing(image, sigma)


def find_local_maxima(ifp, anisotropy):

    try:

        ifp.anytask(gaussian_smoothing, 20/anisotropy, ids='curdisttransf', targetids='smoothed')

        # # If the gaussian smoothing works this will probably work as well...
        # ifp.astype(np.float32, ids='smoothed')
        # ifp.anytask(vigra.analysis.extendedLocalMaxima3D, neighborhood=26, ids='smoothed', targetids='locmax')

    except:

        return False

    return True


def find_large_objects():

    anisotropy = np.array([10, 1, 1])

    folder = '/media/julian/Daten/neuraldata/cremi_2016/'
    file = 'cremi.splA.raw_neurons.crop.crop_10-200-200_110-712-712.disttransf.h5'
    names = ('labels', 'disttransf')
    keys = ('labels', 'disttransf')

    ifp = ImageFileProcessing(
        folder,
        file, asdict=True,
        image_names=names,
        keys=keys)

    ifp.startlogger(filename=None, type='a')

    ifp.addtodict([], 'largeobjects')

    c = 0
    for lblo in ifp.label_bounds_iterator('labels', 'curlabel', ids='disttransf', targetids='curdisttransf',
                                          maskvalue=0, value=0):

        ifp.logging('------------\nCurrent label {} in iteration {}', lblo['label'], c)
        ifp.logging('Bounding box = {}', lblo['bounds'])

        local_maxima_found = find_local_maxima(ifp, anisotropy)

        ifp.logging('Local maxima found: {}', local_maxima_found)

        if local_maxima_found:

            ifp.get_image('largeobjects').append(lblo['label'])

            # if ifp.amax('locmax') != 0:
            #
            #     ifp.write(filename='{0}input_{1}.h5'.format(resultfolder, lblo['label']), ids=('curdisttransf', 'curlabel', 'smoothed'))
            #     ifp.write(filename='{0}paths_over_dist_{1}.h5'.format(resultfolder, lblo['label']), ids=('paths_over_dist',))
            #
            # else:
            #
            #     ifp.logging('No maxima found!')

        else:

            ifp.logging('Maxima detection not successful!')

        c += 1
        # if c == 10:
        #     break

    ifp.write(filename='largeobjects.h5', ids=('largeobjects',))

    ifp.logging('')
    ifp.stoplogger()


def remove_small_objects():

    folder = '/media/julian/Daten/neuraldata/cremi_2016/'
    file = 'cremi.splA.raw_neurons.crop.crop_10-200-200_110-712-712.disttransf.h5'
    names = ('labels',)
    keys = ('labels',)

    ifp = ImageFileProcessing(
        folder,
        file, asdict=True,
        image_names=names,
        keys=keys)

    ifp.startlogger(filename=None)

    # Find all relevant labels
    ifp.addfromfile('{}largeobjects.h5'.format(folder), image_ids=(0,))

    for lbl in ifp.label_image_iterator(
            'labels', 'largeobjects', accumulate=True, labellist=ifp.get_image('largeobjects')):
        ifp.logging('Label {} done!', lbl)

    ifp.getlabel(0, ids='largeobjects', targetids='background')

    ifp.write(filename='largeobjectimage.h5')

    ifp.logging('')
    ifp.stoplogger()


if __name__ == '__main__':

    remove_small_objects()