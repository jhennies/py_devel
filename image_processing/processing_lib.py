
import numpy as np
import scipy
from scipy import ndimage, misc
import vigra
from copy import deepcopy
from vigra import graphs
import math

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
    if type(label) is tuple or type(label) is list:

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


def find_bounding_rect(image, s_=False):

    if image.ndim == 3:

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

        if s_:
            return np.s_[rows.min():rows.max()+1, cols.min():cols.max()+1, bnds.min():bnds.max()+1]
        else:
            return [[rows.min(), rows.max() + 1], [cols.min(), cols.max() + 1], [bnds.min(), bnds.max() + 1]]

    elif image.ndim == 2:

        rows = np.flatnonzero(image.sum(axis=0))
        cols = np.flatnonzero(image.sum(axis=1))

        if s_:
            return np.s_[rows.min():rows.max()+1, cols.min():cols.max()+1]
        else:
            return [[rows.min(), rows.max()+1], [cols.min(), cols.max()+1]]

    else:
        raise TypeError('find_bounding_rect: This number of dimensions ({}) is currently not supported!'.format(image.ndim))


def crop_bounding_rect(image, bounds=None):

    if bounds is None:
        bounds = find_bounding_rect(image)

    if type(bounds) is list:
        return image[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1], bounds[2][0]:bounds[2][1]]
    else:
        return deepcopy(image[bounds])


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


def unique(image, return_counts=False):
    return np.unique(image, return_counts=return_counts)


def gaussian_smoothing(image, sigma, anisotropy=None):
    if anisotropy:
        if type(sigma) is not list and type(sigma) is not tuple and type(sigma) is not np.array:
            sigma = np.array([sigma]*3).astype(np.float32) / anisotropy
        else:
            sigma = np.array(sigma) / anisotropy
    image = image.astype(np.float32)
    return vigra.filters.gaussianSmoothing(image, sigma)


def hessian_of_gaussian_eigenvalues(image, scale, anisotropy=None):
    # if anisotropy:
    #     if type(scale) is not list and type(scale) is not tuple and type(scale) is not np.array:
    #         scale = np.array([scale]*3).astype(np.float32) / anisotropy
    #     else:
    #         scale = np.array(scale) / anisotropy

    image = image.astype(np.float32)
    result = vigra.filters.hessianOfGaussianEigenvalues(image, scale)
    return result


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


def getvaluesfromcoords(image, coordinates):

    values = [image[x[0], x[1], x[2]] for x in coordinates]

    return values


def get_faces_with_neighbors(image, rtrntype=dict):

    faces = rtrntype()

    # --- XY ---
    # w = x + 2*z, h = y + 2*z
    shpxy = (image.shape[0] + 2*image.shape[2], image.shape[1] + 2*image.shape[2])
    xy0 = (0, 0)
    xy1 = (image.shape[2],) * 2
    xy2 = (image.shape[2] + image.shape[0], image.shape[2] + image.shape[1])
    print shpxy, xy0, xy1, xy2

    # xy front face
    xyf = np.zeros(shpxy)
    xyf[xy1[0]:xy2[0], xy1[1]:xy2[1]] = image[:, :, 0]
    xyf[0:xy1[0], xy1[1]:xy2[1]] = np.swapaxes(np.fliplr(image[0, :, :]), 0, 1)
    xyf[xy2[0]:shpxy[0], xy1[1]:xy2[1]] = np.swapaxes(image[-1, :, :], 0, 1)
    xyf[xy1[0]:xy2[0], 0:xy1[1]] = np.fliplr(image[:, 0, :])
    xyf[xy1[0]:xy2[0], xy2[1]:shpxy[1]] = image[:, -1, :]

    # xy back face
    xyb = np.zeros(shpxy)
    xyb[xy1[0]:xy2[0], xy1[1]:xy2[1]] = image[:, :, -1]
    xyb[0:xy1[0], xy1[1]:xy2[1]] = np.swapaxes(image[0, :, :], 0, 1)
    xyb[xy2[0]:shpxy[0], xy1[1]:xy2[1]] = np.swapaxes(np.fliplr(image[-1, :, :]), 0, 1)
    xyb[xy1[0]:xy2[0], 0:xy1[1]] = image[:, 0, :]
    xyb[xy1[0]:xy2[0], xy2[1]:shpxy[1]] = np.fliplr(image[:, -1, :])

    # --- XZ ---
    # w = x + 2*y, h = z + 2*y
    shpxz = (image.shape[0] + 2*image.shape[1], image.shape[2] + 2*image.shape[1])
    xz0 = (0, 0)
    xz1 = (image.shape[1],) * 2
    xz2 = (image.shape[1] + image.shape[0], image.shape[1] + image.shape[2])
    print shpxz, xz0, xz1, xz2

    # xz front face
    xzf = np.zeros(shpxz)
    xzf[xz1[0]:xz2[0], xz1[1]:xz2[1]] = image[:, 0, :]
    xzf[0:xz1[0], xz1[1]:xz2[1]] = np.flipud(image[0, :, :])
    xzf[xz2[0]:shpxz[0], xz1[1]:xz2[1]] = image[-1, :, :]
    xzf[xz1[0]:xz2[0], 0:xz1[1]] = np.fliplr(image[:, :, 0])
    xzf[xz1[0]:xz2[0], xz2[1]:shpxz[1]] = image[:, :, -1]

    # xz back face
    xzb = np.zeros(shpxz)
    xzb[xz1[0]:xz2[0], xz1[1]:xz2[1]] = image[:, -1, :]
    xzb[0:xz1[0], xz1[1]:xz2[1]] = image[0, :, :]
    xzb[xz2[0]:shpxz[0], xz1[1]:xz2[1]] = np.flipud(image[-1, :, :])
    xzb[xz1[0]:xz2[0], 0:xz1[1]] = image[:, :, 0]
    xzb[xz1[0]:xz2[0], xz2[1]:shpxz[1]] = np.fliplr(image[:, :, -1])

    # --- YZ ---
    # w = y + 2*x, h = z + 2*x
    shpyz = (image.shape[1] + 2*image.shape[0], image.shape[2] + 2*image.shape[0])
    yz0 = (0, 0)
    yz1 = (image.shape[0],) * 2
    yz2 = (image.shape[0] + image.shape[1], image.shape[0] + image.shape[2])
    print shpyz, yz0, yz1, yz2

    # yz front face
    yzf = np.zeros(shpyz)
    yzf[yz1[0]:yz2[0], yz1[1]:yz2[1]] = image[0, :, :]
    yzf[0:yz1[0], yz1[1]:yz2[1]] = np.flipud(image[:, 0, :])
    yzf[yz2[0]:shpyz[0], yz1[1]:yz2[1]] = image[:, -1, :]
    yzf[yz1[0]:yz2[0], 0:yz1[1]] = np.swapaxes(np.flipud(image[:, :, 0]), 0, 1)
    yzf[yz1[0]:yz2[0], yz2[1]:shpyz[1]] = np.swapaxes(image[:, :, -1], 0, 1)

    # yz back face
    yzb = np.zeros(shpyz)
    yzb[yz1[0]:yz2[0], yz1[1]:yz2[1]] = image[-1, :, :]
    yzb[0:yz1[0], yz1[1]:yz2[1]] = image[:, 0, :]
    yzb[yz2[0]:shpyz[0], yz1[1]:yz2[1]] = np.flipud(image[:, -1, :])
    yzb[yz1[0]:yz2[0], 0:yz1[1]] = np.swapaxes(image[:, :, 0], 0, 1)
    yzb[yz1[0]:yz2[0], yz2[1]:shpyz[1]] = np.swapaxes(np.flipud(image[:, :, -1]), 0, 1)

    faces['xyf'] = xyf
    faces['xyb'] = xyb
    faces['xzf'] = xzf
    faces['xzb'] = xzb
    faces['yzf'] = yzf
    faces['yzb'] = yzb

    return faces


def shortest_paths(indicator, pairs, bounds=None, logger=None,
                   return_pathim=True, yield_in_bounds=False):

    # Crate the grid graph and shortest path objects
    gridgr = graphs.gridGraph(indicator.shape)
    indicator = indicator.astype(np.float32)
    gridgr_edgeind = graphs.edgeFeaturesFromImage(gridgr, indicator)
    instance = graphs.ShortestPathPathDijkstra(gridgr)

    # Initialize paths image
    if return_pathim:
        pathsim = np.zeros(indicator.shape)
    # Initialize list of path coordinates
    paths = []
    if yield_in_bounds:
        paths_in_bounds = []

    for pair in pairs:

        source = pair[0]
        target = pair[1]

        if logger is not None:
            logger.logging('Calculating path from {} to {}', source, target)

        targetNode = gridgr.coordinateToNode(target)
        sourceNode = gridgr.coordinateToNode(source)

        instance.run(gridgr_edgeind, sourceNode, target=targetNode)
        path = instance.path(pathType='coordinates')
        if path.any():
            # Do not forget to correct for the offset caused by cropping!
            if bounds is not None:
                paths.append(path + [bounds[0].start, bounds[1].start, bounds[2].start])
                if yield_in_bounds:
                    paths_in_bounds.append(path)
            else:
                paths.append(path)

        pathindices = np.swapaxes(path, 0, 1)
        if return_pathim:
            pathsim[pathindices[0], pathindices[1], pathindices[2]] = 1

    if return_pathim:
        if yield_in_bounds:
            return paths, pathsim, paths_in_bounds
        else:
            return paths, pathsim
    else:
        if yield_in_bounds:
            return paths, paths_in_bounds
        else:
            return paths


def split(image, sections, axis=0, result_keys=None, rtrntype=dict):

    shp = list(image.shape)
    # print shp
    #
    # print float(shp[axis]) / 2
    # shp[axis] = round(float(shp[axis]) / 2)

    if float(shp[axis]) / sections != shp[axis] / sections:
        shp[axis] = shp[axis] / sections * sections

        image = image[:shp[0], :shp[1], :shp[2]]

    result = np.split(image, sections, axis=axis)

    if result_keys is None:
        return result
    else:
        if len(result_keys) != len(result):
            raise RuntimeError('processing_lib.split: Number of result keys does not match the number of sections!')
        resultdict = rtrntype()

        for i in xrange(len(result_keys)):

            resultdict[result_keys[i]] = result[i]

        return resultdict


def compute_path_length(path, anisotropy):
    """
    Computes the length of a path

    :param path:
        np.array([[x11, x12, ..., x1n], [x21, x22, ..., x2n], ..., [xm1, xm2, ..., xmn]])
        with n dimensions and m coordinates
    :param anisotropy: [a1, a2, ..., an]
    :return: path length (float)
    """

    pathlen = 0.
    for i in xrange(1, len(path)):

        add2pathlen = 0.
        for j in xrange(0, len(path[0, :])):
            add2pathlen += (anisotropy[j] * (path[i, j] - path[i-1, j])) ** 2

        pathlen += add2pathlen ** (1./2)

    return pathlen
