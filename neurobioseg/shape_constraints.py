from image_processing import ImageFileProcessing
import numpy as np


def calc_taperingtolerance(it, d, type):
    if type == 'd-it':
        return d - it
    elif type == 'it/d':
        return it / d

# TODO: Speed this up by increasing chop-off at each iteration!
# The erosion algorithm:
# ----------------------
def splittingcandidate_by_erosion(ifp, start, stop, tolerance_rel, tolerance_abs, obj):
    for i in xrange(start, stop):
        # print 'i = ' + str(i)

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
                ifp.logging('Object: {0} (abs), {1} (rel)'.format(tt_rel, tt_abs))

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


def tapering_violation_detection(ifp, tolerance_rel=0.5, tolerance_abs=5, pixel_pitch=()):

    for obj in xrange(1, ifp.amax(ids=('labels',))['labels'] + 1):
        pass

        ifp.logging('_____________________________________________________________________')
        ifp.logging('Object label: {}'.format(obj))

        ifp.deepcopy_entry('labels', 'disttransf')

        # Start with one object only
        ifp.getlabel(obj, ids=('disttransf',))
        if ifp.amax(ids=('disttransf',)) == 0:
            ifp.logging('Skiping object {}: Not found.'.format(obj))
            # sys.exit()
            break
        # ifp.write()

        # Distance transform
        ifp.invert_image(ids=('disttransf',))
        ifp.distance_transform(pixel_pitch=pixel_pitch, ids=('disttransf',))
        ifp.write(filename='multicut_segmentation.lbl_' + str(obj) + '.disttransf.h5',
                  dict_ids=('labels', 'disttransf'))

        # ifp.converttodict('disttransf')

        max_disttransf = int(ifp.amax(('disttransf',))['disttransf'])
        ifp.logging('max_disttransf = {}'.format(max_disttransf))
        # print max_disttransf

        erosiondepth = splittingcandidate_by_erosion(ifp, 1, max_disttransf, tolerance_rel, tolerance_abs, obj)

        if erosiondepth is not None:
            ifp.load_h5(im_file='/media/julian/Daten/neuraldata/isbi_2013/data_crop/probabilities_test.h5',
                        im_ids=None, im_names=None, asdict=True, keys=('ws',), append=True)

            ifp.skimage_watershed(markers=ifp.get_image('conncomp'),
                                  mask=ifp.get_image('disttransf'),
                                  ids=('ws',))
            # ifp.anytask(watershed, '', ('ws',), ifp.get_data()['conncomp'], mask=ifp.get_data()['disttransf'] > 0)

            ifp.write(filename='ws.lbl_' + str(obj) + '.i_' + str(erosiondepth) + '.h5')

if __name__ == "__main__":

    ifp = ImageFileProcessing(
        "/media/julian/Daten/neuraldata/isbi_2013/mc_crop_cache/",
        "multicut_segmentation.h5", asdict=True, keys=('labels',))

    tolerance_rel = 0.5
    tolerance_abs = 5
    pixel_pitch = (5, 5, 30)

    ifp.startlogger('/media/julian/Daten/neuraldata/isbi_2013/mc_crop_cache/test.log', type='w')

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
