
import numpy as np
import vigra
from image_processing import ImageProcessing, ImageFileProcessing
import image_processing
from scipy import ndimage
import copy
import sys
from skimage.feature import peak_local_max
from skimage.morphology import watershed

__author__ = 'jhennies'


# def getlabel(img, label):
#     return (img == label).astype(np.uint32)


def distancetransform(img):
    return vigra.filters.distanceTransform(img, pixel_pitch=[5, 5, 30])
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


def delobj(img, label):
    img[img == label] = 0
    return img


def calc_taperingtolerance(it, d, type):
    if type == 'd-it':
        return d - it
    elif type == 'it/d':
        return it / d


if __name__ == "__main__":

    sys.stdout = open('/media/julian/Daten/neuraldata/isbi_2013/mc_crop_cache/log.txt', "a")

    # Parameters
    tapering_tolerance_rel = 0.5
    tapering_tolerance_abs = 5
    # object = 191

    ifp = ImageFileProcessing(
        "/media/julian/Daten/neuraldata/isbi_2013/mc_crop_cache/",
        "multicut_segmentation.h5", asdict=True, keys=('labels',))

    # for obj in [191]:
    for obj in xrange(1, ifp.amax(ids=('labels',))['labels'] + 1):

        sys.stdout = open('/media/julian/Daten/neuraldata/isbi_2013/mc_crop_cache/log.txt', "a")

        print '_____________________________________________________________________'
        print 'Object label: ' + str(obj)

        ifp.deepcopy_entry('labels', 'disttransf')

        if True:

            # Start with one object only
            ifp.getlabel(obj, ids=('disttransf',))
            if ifp.amax(ids=('disttransf',)) == 0:
                print 'Skiping object ' + str(obj) + ': Not found.'
                # sys.exit()
                break
            # ifp.write()

            # Distance transform
            ifp.invert_image(ids=('disttransf',))
            ifp.distance_transform(pixel_pitch=(5,5,30), ids=('disttransf',))
            ifp.write(filename='multicut_segmentation.lbl_' + str(obj) + '.disttransf.h5', dict_ids=('labels', 'disttransf'))

        else:
            ifp = ImageFileProcessing(
                "/media/julian/Daten/neuraldata/isbi_2013/mc_crop_cache/",
                "multicut_segmentation.lbl_" + str(obj) + ".disttransf.h5",
                asdict=True, keys=('disttransf',)
            )

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
        def splittingcandidate_by_erosion(start, stop, ifp):

            for i in xrange(start, stop):
                # print 'i = ' + str(i)

                # Create connected components
                ifp.deepcopy_entry('disttransf', 'conncomp')
                ifp.anytask(binarize, '', ('conncomp',), i)
                ifp.anytask(conncomp, '', ('conncomp',), neighborhood='indirect')

                # print 'Number of objects: ' + str(ifp.amax(('conncomp',))['conncomp'])

                # Check potential candidates according to tapering tolerance
                if ifp.amax(('conncomp',))['conncomp'] > 1:

                    print 'i = ' + str(i)

                    mainarbors = 0

                    for l in xrange(1, ifp.amax(('conncomp',))['conncomp']+1):

                        distinobj = ifp.get_data()['disttransf'][ifp.get_data()['conncomp'] == l]
                        # print distinobj
                        print 'Object ' + str(l) + ': ' + str(np.amax(distinobj) - i) + ' (abs), ' \
                                + str(calc_taperingtolerance(i, np.amax(distinobj), 'it/d')) + ' (rel)'

                        if (calc_taperingtolerance(i, np.amax(distinobj), 'it/d') < tapering_tolerance_rel) \
                                and \
                                (calc_taperingtolerance(i, np.amax(distinobj), 'd-it') > tapering_tolerance_abs):
                            mainarbors += 1
                        else:
                            ifp.anytask(delobj, '', ('conncomp',), l)

                    if mainarbors > 1:

                        ifp.write(filename='tapering_violation.lbl_' + str(obj) + '.i_' + str(i) + '.h5')
                        print 'Tapering violation detected! ' + str(mainarbors) + ' objects found.'

                        return i

            return None

        # ifp.converttodict('disttransf')

        if True:
            max_disttransf = int(ifp.amax(('disttransf',))['disttransf'])
            print max_disttransf

            erosiondepth = splittingcandidate_by_erosion(1, max_disttransf, ifp)

        else:
            i = 29
            ifp = ImageFileProcessing(
                '/media/julian/Daten/neuraldata/isbi_2013/mc_crop_cache/',
                'tapering_violation.lbl_' + str(obj) + '.i_' + str(i) + '.h5',
                asdict=True
            )

        if erosiondepth is not None:
            ifp.load_h5(im_file='/media/julian/Daten/neuraldata/isbi_2013/data_crop/probabilities_test.h5',
                        im_ids=None, im_names=None, asdict=True, keys=('ws',), append=True)

            ifp.anytask(watershed, '', ('ws',), ifp.get_data()['conncomp'], mask=ifp.get_data()['disttransf']>0)

            ifp.write(filename='ws.lbl_' + str(obj) + '.i_' + str(erosiondepth) + '.h5')

        sys.stdout.close()