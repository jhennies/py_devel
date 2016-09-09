
import numpy as np
import vigra
from image_processing import ImageProcessing, ImageFileProcessing
from scipy import ndimage
import copy
import sys


__author__ = 'jhennies'


# def getlabel(img, label):
#     return (img == label).astype(np.uint32)


def distancetransform(img):
    return vigra.filters.distanceTransform(img)
    # return ndimage.distance_transform_edt(img)


def conncomp(img, neighborhood='direct'):
    return vigra.analysis.labelMultiArrayWithBackground(img, neighborhood=neighborhood, background_value=0)


def binarize(img):
    returnimg = copy.deepcopy(img)
    returnimg[returnimg > 1] = 1
    return returnimg


def settozero(img, maxvalue):
    img[img <= maxvalue] = 0
    return img


if __name__ == "__main__":

    # Parameters
    tapering_tolerance = 5
    object = 172

    ifp = ImageFileProcessing(
        "/media/julian/Daten/neuraldata/isbi_2013/mc_crop_cache/",
        "multicut_segmentation.h5", None, 0)

    # Start with one object only
    ifp.getlabel(object)
    if ifp.amax() == 0:
        print 'Terminating scirpt: No object found.'
        sys.exit()
    ifp.write()

    # Get the distance transform inside this object
    ifp.invert_image()
    ifp.anytask(distancetransform, '.disttransf')
    disttransf = copy.deepcopy(ifp.get_image())
    max_disttransf = np.amax(disttransf)

    for i in xrange(1, int(max_disttransf+1)):

        print 'i = ' + str(i)

        # Remove value i from distance transform
        ifp.anytask(settozero, '.' + str(i), i)
        disttransf = copy.deepcopy(ifp.get_image())

        # Get connected components
        binarized = binarize(disttransf)
        conncomps = conncomp(binarized, neighborhood='indirect')
        print 'Number of objects: ' + str(np.amax(conncomps))

        # Check potential candidates according to tapering tolerance
        if np.amax(conncomps) > 1:

            mainarbors = 0

            for l in xrange(1, np.amax(conncomps)+1):

                distinobj = disttransf[conncomps == l]
                # print distinobj
                print 'Maximum distance in object ' + str(l) + ': ' + str(np.amax(distinobj) - i)

                if np.amax(distinobj) - i > tapering_tolerance:
                    mainarbors += 1

            if mainarbors > 1:
                ifp.write()
                print 'Tapering violation detected! ' + str(mainarbors) + ' objects found.'




    # ifp.write()

    # image = ifp.get_image()
    # print image.shape
    #
    # print np.swapaxes(image[500:512, 400:480, 75], 0, 1)