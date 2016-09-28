
from image_processing import ImageProcessing, ImageFileProcessing
import numpy as np


def ifp_boundary_distance_transform(ifp, inid, outid, pixel_pitch=()):

    # ifp.astype(np.float32, ids=inid)
    ifp.set_data_dict({outid: np.zeros(ifp.get_image(inid).shape, dtype=ifp.get_image(inid).dtype)}, append=True)

    # ifp.deepcopy_entry('labels', 'disttransf')
    ifp.set_data_dict({'disttransf': np.zeros(ifp.shape(ids='labels'))}, append=True)

    c = 0
    for lbl in ifp.label_image_iterator(inid, 'currentlabel'):

        ifp.logging('---')
        ifp.logging('lbl {} in iteration {}', lbl, c)

        bounds = ifp.find_bounding_rect(ids='currentlabel')

        ifp.logging('bounds = {}', bounds)
        ifp.crop_bounding_rect(bounds=bounds, ids='currentlabel')

        ifp.distance_transform(pixel_pitch=pixel_pitch, ids='currentlabel', background=False)

        ifp.replace_subimage(ids='disttransf', ids2='currentlabel', bounds=bounds, ignore=0)

        c += 1
        # if c == 10:
        #     break


# # Nope...
# def distance_transform_2ims(imagein, imageout=None, pixel_pitch=(), background=True):
#     vigra.filters.distanceTransform(imagein, out=imageout, pixel_pitch=pixel_pitch, background=background)
#
#
# # Bullshit
# def stretch_image(image, axis, factor):
#     f = [1, 1, 1]
#     f[axis] = factor
#     newimage = np.zeros(np.array(image.shape) * f, dtype=image.dtype)
#
#     for i in xrange(image.shape[axis]):
#         for j in xrange(factor):
#             if axis == 2:
#                 newimage[:, :, i*factor+j] = image[:, :, i]
#             elif axis == 1:
#                 pass
#             elif axis == 0:
#                 newimage[i*factor+j, :, :] = image[i, :, :]
#
#     print newimage.shape
#     return newimage
#
# # Nope...
# def boundary_distance_transform_anisotropic(image):
#     # image.astype(np.float16)
#     # image = np.zeros((100, 100, 100), dtype=np.float32)
#     # return vigra.sampling.resizeVolumeSplineInterpolation(image, shape=(100, 100, 100))
#
#     stretch_image(image, 0, 10)
#
#     return image


if __name__ == '__main__':

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

    ifp.startlogger(filename='/media/julian/Daten/neuraldata/cremi_2016/disttransf.log', type='a')

    ifp.logging('keys = {}', ifp.get_data().keys())

    if crop:
        ifp.crop([10, 200, 200], [110, 712, 712])

    ifp_boundary_distance_transform(ifp, 'labels', 'disttransf', pixel_pitch=anisotropy)


    ifp.logging('ifp.shape() = {}', ifp.shape())

    ifp.write(filename='cremi.splA.raw_neurons.crop.crop_10-200-200_110-712-712.disttransf.h5', ids=('labels', 'disttransf'))

    ifp.logging('')

    ifp.stoplogger()
