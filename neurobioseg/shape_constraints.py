from image_processing import ImageFileProcessing
import numpy as np
from skimage.feature import peak_local_max
import vigra
import scipy

def calc_taperingtolerance(it, d, type):
    if type == 'd-it':
        return d - it
    elif type == 'it/d':
        return it / d


# Doesn't work: its not local minima but saddle points!
# Possible approach: A watershed based on the distancetransform splits at saddle points!
#
# def splittingcandidate_by_localminima(ifp, start, stop, step, tolerance_rel, tolerance_abs, obj):
#
#     ifp.deepcopy_entry('disttransf', 'localmin')
#     ifp.invert_image(ids=('localmin',))
#
#     # ifp.anytask(peak_local_max, '', ('localmin',), min_distance=5)
#
#     ifp.anytask(vigra.filters.discRankOrderFilter, '', ('localmin',), 5, 1)
#     ifp.anytask(vigra.analysis.extendedLocalMaxima3D, '', ('localmin',), neighborhood=26)
#     print ifp.amax(ids=('localmin',))
#
#     ifp.write(filename='localmin.{}.h5'.format(obj), ids=('localmin', 'disttransf'))
#
#
#     return None

# Doesn't work: I'm missing information on the saddle point inbetween two maxima.
# The currently best approach remains the splitting_candidate_by_erosion
# def splitting_by_localmax(ifp, obj):
#
#     # IDEA: Equivalent algorithm to splittingcandidate_by_erosion
#     #   Using local maxima as candidates and using the same constraints to exclude or keep them
#
#     ifp.deepcopy_entry('disttransf', 'ws')
#     ifp.invert_image(ids=('ws',))
#     ifp.deepcopy_entry('disttransf', 'locmax')
#     # ifp.anytask(peak_local_max, '', ('locmax',), min_distance=5)
#
#     # ifp.anytask(vigra.filters.discRankOrderFilter, '', ('locmax',), 3, 1)
#     ifp.anytask(scipy.ndimage.maximum_filter, '', ('locmax',), size=(10, 10, 10))
#     ifp.anytask(vigra.analysis.extendedLocalMaxima3D, '', ('locmax',), neighborhood=26)
#     ifp.conncomp('indirect', 0, ids=('locmax',))
#     print ifp.amax(ids=('locmax',))
#
#     ifp.skimage_watershed(markers=ifp.get_image('locmax'),
#                                   mask=ifp.get_image('disttransf')>0,
#                                   ids=('ws',))
#     # ifp.anytask(watershed, '', ('ws',), ifp.get_data()['conncomp'], mask=ifp.get_data()['disttransf'] > 0)
#
#     ifp.write(filename='ws.{}.h5'.format(obj))
#
#
#     return None


# The erosion algorithm:
# ----------------------
def splittingcandidate_by_erosion(ifp, start, stop, step, tolerance_rel, tolerance_abs, obj):

    for i in xrange(start, stop, step):

        # print 'i = ' + str(i)
        # ifp.logging("i = {}".format(i))
        # Create connected components
        ifp.deepcopy_entry('disttransf', 'conncomp')
        ifp.binarize(i, type='l', ids=('conncomp',))
        # ifp.anytask(binarize, '', ('conncomp',), i)
        ifp.astype(np.uint32, ids=('conncomp',))
        ifp.conncomp(neighborhood='indirect', background_value=0, ids=('conncomp',))
        # ifp.anytask(conncomp, '', ('conncomp',), neighborhood='indirect')

        # print 'Number of objects: ' + str(ifp.amax(('conncomp',))['conncomp'])

        # Check potential candidates according to tapering tolerance
        if ifp.amax(('conncomp',))['conncomp'] > 1:

            ifp.logging("i = {}".format(i))

            mainarbors = 0

            for l in xrange(1, ifp.amax(('conncomp',))['conncomp'] + 1):

                distinobj = ifp.get_data()['disttransf'][ifp.get_data()['conncomp'] == l]
                # print distinobj
                # print 'Object ' + str(l) + ': ' + str(np.amax(distinobj) - i) + ' (abs), ' \
                #       + str(calc_taperingtolerance(i, np.amax(distinobj), 'it/d')) + ' (rel)'
                tt_abs = calc_taperingtolerance(i, np.amax(distinobj), 'd-it')
                tt_rel = calc_taperingtolerance(i, np.amax(distinobj), 'it/d')
                ifp.logging('Object: {0} (abs), {1} (rel)'.format(tt_abs, tt_rel))

                if (tt_rel < tolerance_rel) and (tt_abs > tolerance_abs):
                    mainarbors += 1
                else:
                    # ifp.anytask(delobj, '', ('conncomp',), l)
                    ifp.filter_values(l, type='eq', setto=0, ids=('conncomp',))

            if mainarbors > 1:
                ifp.write(filename='tapering_violation.lbl_' + str(obj) + '.i_' + str(i) + '.h5')
                ifp.logging('Tapering violation detected! {} obects found.'.format(mainarbors))
                # print 'Tapering violation detected! ' + str(mainarbors) + ' objects found.'

                return i

    return None


def split_candidate(ifp, obj, type='wsonprobs'):

    if type == 'wsonprobs':
        ifp.load_h5(im_file='/media/julian/Daten/neuraldata/isbi_2013/data_crop/probabilities_test.h5',
                    im_ids=None, im_names=None, asdict=True, keys=('ws',), append=True)

        ifp.skimage_watershed(markers=ifp.get_image('conncomp'),
                              mask=ifp.get_image('disttransf'),
                              ids=('ws',))
        # ifp.anytask(watershed, '', ('ws',), ifp.get_data()['conncomp'], mask=ifp.get_data()['disttransf'] > 0)
    elif type == 'wsondisttransf':
        ifp.deepcopy_entry('disttransf', 'ws')
        ifp.invert_image(('ws',))
        ifp.skimage_watershed(markers=ifp.get_image('conncomp'),
                              mask=ifp.get_image('disttransf'),
                              ids=('ws',))


    ifp.write(filename='split.lbl_{}.h5'.format(obj))


def tapering_violation_detection(ifp, tolerance_rel=0.5, tolerance_abs=5, pixel_pitch=()):

    # TODO: Write function to find a list of present labels
    # TODO: Or convert label image to contain all labels

    for obj in xrange(1, int(ifp.amax(ids=('labels',))['labels'] + 1)):

        ifp.logging('_____________________________________________________________________')
        ifp.logging('Object label: {}'.format(obj))

        ifp.deepcopy_entry('labels', 'disttransf')

        # Start with one object only
        ifp.getlabel(obj, ids=('disttransf',))
        if ifp.amax(ids=('disttransf',))['disttransf'] > 0:

            # # -------------------------------
            # # Write the selected label image
            ifp.write(filename='getlabel.{}.h5'.format(obj), ids=('disttransf',))

            # Distance transform
            ifp.invert_image(ids=('disttransf',))
            ifp.distance_transform(pixel_pitch=pixel_pitch, ids=('disttransf',))

            # # ------------------------------
            # # Write the distance transform
            # ifp.write(filename='multicut_segmentation.lbl_' + str(obj) + '.disttransf.h5',
            #           ids=('labels', 'disttransf'))

            # ifp.converttodict('disttransf')

            max_disttransf = int(ifp.amax(('disttransf',))['disttransf'])
            ifp.logging('max_disttransf = {}'.format(max_disttransf))
            # print max_disttransf

            step = int(max_disttransf / 24) + 1
            tolerance_abs_in = tolerance_abs * step
            erosiondepth = splittingcandidate_by_erosion(ifp, 1, max_disttransf, step, tolerance_rel, tolerance_abs_in,
                                                         obj)
            # erosiondepth = splitting_by_localmax(ifp, obj)

            if erosiondepth is not None:
                split_candidate(ifp, obj, type='wsondisttransf')

            ifp.write(filename='object_{}.h5'.format(obj))

        else:

            ifp.logging('Skiping object {}: Not found.'.format(obj))
            # sys.exit()




if __name__ == "__main__":

    folder = '/media/julian/Daten/neuraldata/cremi_2016/'
    file = 'cremi.splA.raw_neurons.crop.h5'
    names = ('neuron_ids',)
    keys = ('labels',)

    tolerance_rel = 0.5
    tolerance_abs = 5
    pixel_pitch = (1, 1, 10)

    ifp = ImageFileProcessing(
        folder,
        file, asdict=True,
        image_names=names,
        keys=keys)

    ifp.startlogger('{}tapering.log'.format(folder), type='w')

    ifp.logging("######################################################")
    ifp.logging('# Starting tapering_violation_detection with:        #')
    ifp.logging("#    tolerance_rel = {0: <31} #".format(tolerance_rel))
    ifp.logging("#    tolerance_abs = {0: <31} #".format(tolerance_abs))
    ifp.logging("#    pixel_pitch = {0: <34}#".format(pixel_pitch))
    ifp.logging("######################################################\n")

    tapering_violation_detection(ifp,
                                 tolerance_rel=tolerance_rel,
                                 tolerance_abs=tolerance_abs,
                                 pixel_pitch=pixel_pitch)

    ifp.logging('######################################################')
    ifp.logging('# tapering_violation_detection finished successfully #')
    ifp.logging("######################################################\n")

    ifp.stoplogger()
