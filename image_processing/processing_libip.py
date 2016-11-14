
import numpy as np
from skimage import morphology
import vigra
import processing_lib as lib
from hdf5_image_processing import Hdf5ImageProcessingLib as IPL

__author__ = 'jhennies'


def filter_small_objects(ipl, key, threshold, tkey=None, relabel=False):
    """
    Filters small objects
    :param ipl:
    :param key:
    :param threshold:
    :param relabel:
    :return:
    """

    ipl.logging('Filtering objects smaller than {} voxels...', threshold)

    unique, counts = np.unique(ipl[key], return_counts=True)
    ipl.logging('Found objects: {}\nCorresponding sizes: {}', unique, counts)

    ipl.logging('Removing theses objects: {}', unique[counts <= threshold])

    # With accumulate set to True, this iterator does everything we need:
    # Each label with a count larger than size_exclusion is added to lblim which is initialized as np.zeros(...)
    for lbl, lblim in ipl.label_image_iterator(key=key,
                                               labellist=unique[counts > threshold],
                                               accumulate=True, relabel=relabel):
        ipl.logging('---\nIncluding label {}', lbl)

    if tkey is None:
        ipl[key] = lblim
    else:
        ipl[tkey] = lblim

    ipl.logging('... Done filtering small objects!')

    return ipl


def boundary_disttransf(ipl, key, tkey, anisotropy=[1, 1, 1]):
    """
    Computes the boundary distancetransform for an object within a Hdf5ImageProcessingLib instance

    :param ipl: Hdf5ImageProcessingLib instance containing the image
    :param key: Key of the original image
    :param tkey: Key of the target image
    :param anisotropy: Specify the anisotropy of the image data, default = [1, 1, 1]
    :return: nothing (the result is stored within the Hdf5ImageProcessing instance)
    """

    ipl.logging('Computing boundary distance transform for key = {}', key)
    # Boundary distance transform
    # a) Boundaries
    ipl.logging('Finding boundaries ...')
    ipl.pixels_at_boundary(
        axes=(np.array(anisotropy).astype(np.float32) ** -1).astype(np.uint8),
        keys=key,
        tkeys=tkey
    )
    ipl.astype(np.float32, keys=tkey)

    # b) Distance transform
    ipl.logging('Computing distance transform on boundaries ...')
    ipl.distance_transform(
        pixel_pitch=anisotropy,
        background=True,
        keys=tkey
    )
    
    
def compute_faces(ipl, keys, tkeys):
    
    ipl.logging('Computing faces ...')
    ipl.get_faces_with_neighbors(keys=keys, tkeys=tkeys)

    # startpoints = ipl['faces', keys[0]].keys()
    additionalinfo = IPL()
    for key in keys:
        shp = ipl[key].shape
        startpoints = {'xyf': shp[2],
                       'xyb': shp[2],
                       'xzf': shp[1],
                       'xzb': shp[1],
                       'yzf': shp[0],
                       'yzb': shp[0]}
        areas = {'xyf': np.s_[shp[2]:shp[2] + shp[0], shp[2]:shp[2] + shp[1]],
                 'xyb': np.s_[shp[2]:shp[2] + shp[0], shp[2]:shp[2] + shp[1]],
                 'xzf': np.s_[shp[1]:shp[1] + shp[0], shp[1]:shp[1] + shp[2]],
                 'xzb': np.s_[shp[1]:shp[1] + shp[0], shp[1]:shp[1] + shp[2]],
                 'yzf': np.s_[shp[0]:shp[0] + shp[1], shp[0]:shp[0] + shp[2]],
                 'yzb': np.s_[shp[0]:shp[0] + shp[1], shp[0]:shp[0] + shp[2]]}

        additionalinfo[key, 'startpoints'] = startpoints
        additionalinfo[key, 'areas'] = areas

    return additionalinfo
    
    
def find_border_centroids(ipl, keys, areas, imkey, disttransfkey, resultkey):
    """
    :param ipl:
    :param keys:
    :param areas: supply area in the format area=np.s_[numpy indexing], i.e. area=np.s_[:,:,:] for a full 3d image
            Note that this affects only the determination of which labels are iterated over, when labellist is supplied
            this parameter has no effect
    :param imkey:
    :param disttransfkey:
    :param resultkey:
    :return:
    """

    for k, bounds in keys.iteritems():

        # bounds = (shp[0],) * 2
        for lbl, lblim in ipl['faces', imkey].label_image_iterator(key=k, background=0, area=areas[k]):

            ipl.logging('---\nLabel {} found in image {}', lbl, k)

            # Avoid very small artifacts
            lblim = morphology.opening(lblim)

            # Connected component analysis to detect when a label touches the border multiple times
            conncomp = vigra.analysis.labelImageWithBackground(lblim.astype(np.uint32), neighborhood=8, background_value=0)

            for l in np.unique(conncomp):
                # Ignore background
                if l == 0: continue

                # Get the current label object
                curobj = conncomp == l

                # Get disttancetransf of the object
                curdist = np.array(ipl['faces', disttransfkey, k])
                curdist[curobj == False] = 0

                # Detect the global maximum of this object
                amax = np.amax(curdist)
                curdist[curdist < amax] = 0
                curdist[curdist > 0] = lbl
                # Only one pixel is allowed to be selected
                bds = lib.find_bounding_rect(curdist)
                centroid = (int((bds[1][0] + bds[1][1]-1) / 2), int((bds[0][0] + bds[0][1]-1) / 2))

                # Now translate the calculated centroid to the position within the orignial 3D volume
                centroidm = (centroid[0] - bounds, centroid[1] - bounds)
                # ipl.logging('centroidxy = {}', centroidm)
                # Set the pixel
                try:
                    if centroidm[0] < 0 or centroidm[1] < 0:
                        raise IndexError
                    else:
                        if k == 'xyf':
                            ipl[resultkey][centroidm[0], centroidm[1], 0] = lbl
                        elif k == 'xyb':
                            ipl[resultkey][centroidm[0], centroidm[1], -1] = lbl
                        elif k == 'xzf':
                            ipl[resultkey][centroidm[0], 0, centroidm[1]] = lbl
                        elif k == 'xzb':
                            ipl[resultkey][centroidm[0], -1, centroidm[1]] = lbl
                        elif k == 'yzf':
                            ipl[resultkey][0, centroidm[0], centroidm[1]] = lbl
                        elif k == 'yzb':
                            ipl[resultkey][-1, centroidm[0], centroidm[1]] = lbl
                except IndexError:
                    pass