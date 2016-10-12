
import numpy as np
import scipy
from scipy import ndimage, misc
import vigra

__author__ = 'jhennies'


# _____________________________________________________________________________________________
# The image processing library

def invert_image(image):
    return np.amax(image) - image


def swapaxes(image, axis1, axis2):
    return np.swapaxes(image, axis1, axis2)


def rollaxis(image, axis, start=0):
    return np.rollaxis(image, axis, start)


def resize(image, zoom, mode):
    # self._image = vigra.sampling.resizeImageNoInterpolation(self._image, shape=shape)
    print 'Resizing with mode = ' + mode
    return scipy.ndimage.interpolation.zoom(image, zoom, mode=mode, order=0)
    # scipy.misc.imresize(self._image, shape, interp='nearest')


def resize_z_nearest(image, z):
    img = image
    newimg = np.zeros((img.shape[0], img.shape[1], z))
    for i in xrange(0, img.shape[1]):
        t = img[:, i, :]
        # print img.shape
        # print t.shape
        # t = np.swapaxes(t, 1, 2)
        misc.imresize(t, z, interp='nearest')
        # t = np.swapaxes(t, 1, 2)

        newimg[:, i, :] = t

    return newimg


def getlabel(image, label):
    if type(label) is tuple:

        lblim = np.zeros(image.shape, dtype=image.dtype)
        for lbl in label:
            lblim[image == lbl] = lbl

        return lblim

    else:

        return np.array((image == label)).astype(np.uint32)


def amax(image):
    return np.amax(image)


def astype(image, dtype):
    return image.astype(dtype=dtype)


def distance_transform(image, pixel_pitch=(), background=True):
    return vigra.filters.distanceTransform(image, pixel_pitch=pixel_pitch, background=background)


def filter_values(image, value, type='se', setto=0):

    if type == 's':
        image[image < value] = setto
    elif type == 'se':
        image[image <= value] = setto
    elif type == 'eq':
        image[image == value] = setto
    elif type == 'le':
        image[image >= value] = setto
    elif type == 'l':
        image[image > value] = setto
    elif type == 'ne':
        image[image != value] = setto

    return image


def binarize(image, value, type='l'):

    returnimg = np.array()
    if type == 's':
        returnimg = (image < value)
    elif type == 'se':
        returnimg = (image <= value)
    elif type == 'eq':
        returnimg = (image == value)
    elif type == 'le':
        returnimg = (image >= value)
    elif type == 'l':
        returnimg = (image > value)

    return returnimg


def conncomp(image, neighborhood='direct', background_value=0):
    return vigra.analysis.labelMultiArrayWithBackground(image, neighborhood=neighborhood, background_value=background_value)


def crop(image, start, stop):
    return image[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]


def shape(image):
    return image.shape


def power(image, value):
    return np.power(image, value)


def mult(image, value):
    return np.multiply(image, value)


def mult2im(image1, image2):
    return image1 * image2


def add(image, value):
    return image + value


def add2im(image1, image2):
    return image1 + image2


def concatenate(image1, image2):
    return [image1, image2]


def find_bounding_rect(image):

    # print 'np.amax(image) = {}'.format(np.amax(image))
    # print 'image.sum(axis=0) = {}'.format(image.sum(axis=0))
    # print 'image.sum(axis=0).sum(axis=0) = {}'.format(image.sum(axis=0).sum(axis=0))

    # Bands
    bnds = np.flatnonzero(image.sum(axis=0).sum(axis=0))
    # Rows
    rows = np.flatnonzero(image.sum(axis=1).sum(axis=1))
    # Columns
    cols = np.flatnonzero(image.sum(axis=2).sum(axis=0))

    # print 'image.shape = {}'.format(image.shape)
    # print 'bnds = {}'.format(bnds)
    # print 'rows = {}'.format(rows)
    # print 'cols = {}'.format(cols)

    return [[rows.min(), rows.max()+1], [cols.min(), cols.max()+1], [bnds.min(), bnds.max()+1]]


def crop_bounding_rect(image, bounds=None):

    if bounds is None:
        bounds = find_bounding_rect(image)

    return image[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1], bounds[2][0]:bounds[2][1]]


def replace_subimage(image, rplimage, position=None, bounds=None, ignore=None):

    # Make sure bounds is not None
    if bounds is None and position is None:
        bounds = ((0, 0), (0, 0), (0, 0))
    if bounds is None and position is not None:
        bounds = ((position[0], rplimage.shape[0]), (position[1], rplimage.shape[1]), (position[2], rplimage.shape[2]))

    if ignore is None:
        image[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1], bounds[2][0]:bounds[2][1]] = rplimage
    else:
        image[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1], bounds[2][0]:bounds[2][1]][rplimage != ignore] = rplimage[rplimage != ignore]

    return image


def mask_image(image, mask, maskvalue=False, value=0):
    image[mask == maskvalue] = value
    return image


def unique(image):
    return np.unique(image)


def gaussian_smoothing(image, sigma):
    return vigra.filters.gaussianSmoothing(image, sigma)


def extended_local_maxima(image, neighborhood=26):
    image = image.astype(np.float32)
    return vigra.analysis.extendedLocalMaxima3D(image, neighborhood=neighborhood)


def pixels_at_boundary(image, axes=[1, 1, 1]):

    return axes[0] * ((np.concatenate((image[(0,),:,:], image[:-1,:,:]))
                      - np.concatenate((image[1:,:,:], image[(-1,),:,:]))) != 0) \
        + axes[1] * ((np.concatenate((image[:,(0,),:], image[:,:-1,:]), 1)
                      - np.concatenate((image[:,1:,:], image[:,(-1,),:]), 1)) != 0) \
        + axes[2] * ((np.concatenate((image[:,:,(0,)], image[:,:,:-1]), 2)
                      - np.concatenate((image[:,:,1:], image[:,:,(-1,)]), 2)) != 0)

    # imxp = copy.deepcopy(image)
    # imxp[1:,:,:] = image[:-1,:,:]
    # imxm = copy.deepcopy(image)
    # imxm[:-1,:,:] = image[1:,:,:]

    # imyp = copy.deepcopy(image)
    # imyp[:,1:,:] = image[:,:-1,:]
    # imym = copy.deepcopy(image)
    # imym[:,:-1,:] = image[:,1:,:]
    # imzp = copy.deepcopy(image)
    # imzp[:,:,1:] = image[:,:,:-1]
    # imzm = copy.deepcopy(image)
    # imzm[:,:,:-1] = image[:,:,1:]

    # return axes[0] * ((imxp - imxm) != 0) + axes[1] * ((imyp - imym) != 0) + axes[2] * ((imzp - imzm) != 0)
    # return ((imyp - imym) != 0) + ((imzp - imzm) != 0)


def positions2value(image, coordinates, value):
    image[coordinates[0], coordinates[1], coordinates[2]] = value
    return image
