
import numpy as np
import vigra
from image_processing import ImageProcessing, ImageFileProcessing
from scipy import ndimage
import copy
import sys
from skimage.feature import peak_local_max

__author__ = 'jhennies'


# def getlabel(img, label):
#     return (img == label).astype(np.uint32)


def distancetransform(img):
    return vigra.filters.distanceTransform(img)
    # return ndimage.distance_transform_edt(img)


def conncomp(img, neighborhood='direct'):
    return vigra.analysis.labelMultiArrayWithBackground(img, neighborhood=neighborhood, background_value=0)


def binarize(img, value):
    # returnimg = copy.deepcopy(img)
    returnimg = (img > value).astype(np.uint32)

    return returnimg


def settozero(img, maxvalue):
    img[img <= maxvalue] = 0
    return img


if __name__ == "__main__":

    # Parameters
    tapering_tolerance = 5
    object = 191

    if False:
        ifp = ImageFileProcessing(
            "/media/julian/Daten/neuraldata/isbi_2013/mc_crop_cache/",
            "multicut_segmentation.h5", None, 0)

        # Start with one object only
        ifp.getlabel(object)
        if ifp.amax() == 0:
            print 'Terminating scirpt: No object found.'
            sys.exit()
        ifp.write()

        # Distance transform
        ifp.invert_image()
        ifp.resize([1, 1, 5], 'nearest')
        ifp.anytask(distancetransform, '.disttransf')
        ifp.resize([1, 1, 0.2], 'nearest')
        ifp.write()

    else:
        ifp = ImageFileProcessing(
            "/media/julian/Daten/neuraldata/isbi_2013/mc_crop_cache/",
            "multicut_segmentation.lbl_" + str(object) + ".inv.resize.disttransf.resize.h5", None, 0)

    # #########################################################################################
    # # The local maximum algorithm: Fast but imprecise
    # # -------------------------------------------------
    # # Using the vigra max filter and local maxima
    # # Yet it somehow won't do the trick
    # ifp.anytask(vigra.filters.discRankOrderFilter, '.maxfilt', tapering_tolerance, 1)
    # # ifp.write()
    # ifp.anytask(vigra.analysis.extendedLocalMaxima3D, '.locmax', neighborhood=26)
    #
    # # # Lets try the skimage library
    # # # ifp.anytask(ndimage.maximum_filter, size=20, mode='constant')
    # # ifp.anytask(peak_local_max, '.locmax_skimage', min_distance=tapering_tolerance)
    # # # ifp.write()
    # # print ifp.get_image().shape
    #
    # locmaxs = ifp.get_image()
    # print len(locmaxs[locmaxs > 0])
    # conncomps = conncomp(locmaxs, neighborhood='indirect')
    # print 'Number of objects: ' + str(np.amax(conncomps))

    # ifp.write()

    # #########################################################################################
    # The erosion algorithm:
    # ----------------------
    def splitting_by_erosion(start, stop, ifp):

        for i in xrange(start, stop):
            print 'i = ' + str(i)

            # Create connected components
            ifp.deepcopy('disttransf', 'conncomp')
            ifp.anytask(binarize, '', ('conncomp',), i)
            ifp.anytask(conncomp, '', ('conncomp',), neighborhood='indirect')

            print 'Number of objects: ' + str(ifp.amax(('conncomp',))['conncomp'])

            # Check potential candidates according to tapering tolerance
            if ifp.amax(('conncomp',))['conncomp'] > 1:

                mainarbors = 0

                for l in xrange(1, ifp.amax(('conncomp',))['conncomp']+1):

                    distinobj = ifp.get_image()['disttransf'][ifp.get_image()['conncomp'] == l]
                    # print distinobj
                    print 'Maximum distance in object ' + str(l) + ': ' + str(np.amax(distinobj) - i)

                    if np.amax(distinobj) - i > tapering_tolerance:
                        mainarbors += 1

                if mainarbors > 1:

                    # ifp.addtoname('.violation.step_' + str(i))
                    ifp.write(filename='tapering_violation.lbl_' + str(object) + '.i_' + str(i) + '.h5')
                    print 'Tapering violation detected! ' + str(mainarbors) + ' objects found.'

    ifp.converttodict('disttransf')

    max_disttransf = int(ifp.amax(('disttransf',))['disttransf'])
    print max_disttransf

    splitting_by_erosion(1, max_disttransf, ifp)

    sys.exit()

