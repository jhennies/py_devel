
import numpy as np
from skimage import morphology
import vigra
import processing_lib as lib
from hdf5_image_processing import Hdf5ImageProcessingLib as IPL
import random
from vigra import graphs

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


def merge_adjacent_objects(
        ipl, key, numberbysize=0, numberbyrandom=0, seed=None,
        targetnames=('largeobjm', 'mergeids_small', 'mergeids_random', 'change_hash'),
        algorithm='standard'
):
    """
    :param ipl:

    ipl.get_params():

        merge_adjacent_objects
            seed
            numberbysize
            numberbyrandom

        largeobjmnames
            - 'largeobj_merged'
            - 'mergeids_small'
            - 'mergeids_random'
            - 'mergeids_all'
            - 'change_hash'

    :param key: the source key for calculation
    """

    # This removes the redundancy from the hash
    def reduce_hash(hash):
        br = False
        for k, v in hash.iteritems():
            for x in v:
                if x != k:
                    if x in hash.keys():
                        del hash[x]
                        reduce_hash(hash)
                        br = True
                        break
                    else:
                        br = False
            if br:
                break

    if algorithm == 'pairs':

        # Get only the relevant labels
        labels = lib.unique(ipl[key])
        ipl.logging('labels = {}', labels)

        # Seed the randomize function
        random.seed(seed)

        ipl.astype(np.uint32, keys=key)
        (grag, rag) = graphs.gridRegionAdjacencyGraph(ipl[key], ignoreLabel=0)
        edge_ids = rag.edgeIds()

        merge_ids = []
        used_ids = ()

        # Calculate the merges such that only pairs are found
        for i in xrange(0, numberbyrandom):

            c = 0
            finished = False
            while not finished:

                edge_id = random.choice(edge_ids)
                uid = rag.uId(rag.edgeFromId(edge_id))
                vid = rag.vId(rag.edgeFromId(edge_id))

                c += 1
                if c > 50:
                    ipl.logging('merge_adjacent_objects: Warning: Not finding any more pairs!')
                    finished = True

                elif uid not in used_ids and vid not in used_ids:
                    finished = True
                    used_ids += (uid, vid)
                    merge_ids.append((uid, vid))

        ipl.logging('Label IDs randomly selected for merging: {}', merge_ids)

        # Sort
        merge_ids = [sorted(x) for x in merge_ids]
        merge_ids = sorted(merge_ids)
        ipl.logging('merge_ids = {}', merge_ids)

        # Store this for later use
        ipl[targetnames[2]] = merge_ids
        ipl[targetnames[3]] = merge_ids

        # # Create change hash list
        # change_hash = IPL(data=dict(zip(np.unique(merge_ids), [[x, ] for x in np.unique(merge_ids)])))
        # for i in xrange(0, 3):
        #     prev_change_hash = IPL(data=change_hash)
        #     for x in merge_ids:
        #         ipl.logging('Adding {} and {}', *x)
        #         change_hash[x[0]] += change_hash[x[1]]
        #         change_hash[x[0]] = list(np.unique(change_hash[x[0]]))
        #         change_hash[x[1]] += change_hash[x[0]]
        #         change_hash[x[1]] = list(np.unique(change_hash[x[1]]))
        #
        # reduce_hash(change_hash)
        # # Change the list in the hash to np-arrays for better storage in h5 files
        # for k, v in change_hash.iteritems():
        #     change_hash[k] = np.array(v)
        # # And now we have a perfect change list which we just need to iterate over and change the labels in the image
        # ipl.logging('change_hash after change:')

        us = [x[0] for x in merge_ids]
        change_hash = IPL(data=dict(zip(us, merge_ids)))

        ipl.logging('change_hash: {}', change_hash)
        ipl[targetnames[4]] = change_hash

    elif algorithm == 'standard':

        # Get only the relevant labels
        labels = lib.unique(ipl[key])
        ipl.logging('labels = {}', labels)

        # Seed the randomize function
        random.seed(seed)

        ipl.astype(np.uint32, keys=key)
        (grag, rag) = graphs.gridRegionAdjacencyGraph(ipl[key], ignoreLabel=0)
        edge_ids = rag.edgeIds()
        # ipl.logging('Edge ids: {}', edge_ids)

        # Type 1:
        # Select edges by size (smallest edges)
        ipl.logging('Number of edgeLengths = {}', len(rag.edgeLengths()))
        edgelen_ids = dict(zip(edge_ids, rag.edgeLengths()))
        # ifp.logging('edgelen_ids = {}', edgelen_ids)
        sorted_edgelens = np.sort(rag.edgeLengths())
        #
        smallest_merge_lens = sorted_edgelens[0:numberbysize]
        ipl.logging('Lengths selected for merging: {}', smallest_merge_lens)
        #
        smallest_merge_ids = []
        for x in smallest_merge_lens:
            edge_id = edgelen_ids.keys()[edgelen_ids.values().index(x)]
            smallest_merge_ids.append(edge_id)
            edgelen_ids.pop(edge_id)
        #
        edge_ids = edgelen_ids.keys()
        ipl.logging('Edge IDs selected for merging due to size: {}', smallest_merge_ids)

        # Type 2:
        # Randomly choose edges
        random_merge_ids = random.sample(edge_ids, numberbyrandom)
        ipl.logging('Edge IDs randomly selected for merging: {}', random_merge_ids)

        # Now get the label ids
        smallest_merge_labelids_u = [rag.uId(rag.edgeFromId(x)) for x in smallest_merge_ids]
        smallest_merge_labelids_v = [rag.vId(rag.edgeFromId(x)) for x in smallest_merge_ids]
        smallest_merge_labelids = list(zip(smallest_merge_labelids_u, smallest_merge_labelids_v))
        random_merge_labelids_u = [rag.uId(rag.edgeFromId(x)) for x in random_merge_ids]
        random_merge_labelids_v = [rag.vId(rag.edgeFromId(x)) for x in random_merge_ids]
        random_merge_labelids = list(zip(random_merge_labelids_u, random_merge_labelids_v))
        ipl.logging('Label IDs selected for merging by size: {}', smallest_merge_labelids)
        ipl.logging('Label IDs randomly selected for merging: {}', random_merge_labelids)

        # Concatenate
        all_merge_labelids = smallest_merge_labelids + random_merge_labelids
        # Sort
        ipl.logging('all_merge_labelids = {}', all_merge_labelids)
        all_merge_labelids = [sorted(x) for x in all_merge_labelids]
        all_merge_labelids = sorted(all_merge_labelids)
        ipl.logging('all_merge_labelids = {}', all_merge_labelids)

        # Store this for later use
        ipl[targetnames[1]] = smallest_merge_labelids
        ipl[targetnames[2]] = random_merge_labelids
        ipl[targetnames[3]] = all_merge_labelids

        # Create change hash list
        change_hash = IPL(data=dict(zip(np.unique(all_merge_labelids), [[x,] for x in np.unique(all_merge_labelids)])))
        for i in xrange(0, 3):
            prev_change_hash = IPL(data=change_hash)
            for x in all_merge_labelids:
                ipl.logging('Adding {} and {}', *x)
                change_hash[x[0]] += change_hash[x[1]]
                change_hash[x[0]] = list(np.unique(change_hash[x[0]]))
                change_hash[x[1]] += change_hash[x[0]]
                change_hash[x[1]] = list(np.unique(change_hash[x[1]]))
        # This removes the redundancy from the hash
        reduce_hash(change_hash)
        # Change the list in the hash to np-arrays for better storage in h5 files
        for k, v in change_hash.iteritems():
            change_hash[k] = np.array(v)
        # And now we have a perfect change list which we just need to iterate over and change the labels in the image
        ipl.logging('change_hash after change:')
        ipl.logging(change_hash)
        ipl[targetnames[4]] = change_hash

    # Create the merged image
    # ipl.deepcopy_entry('largeobj', targetnames[0])
    ipl.rename_entry(key, targetnames[0])
    for k, v in change_hash.iteritems():
        for x in v:
            if x != k:
                ipl.logging('Setting {} to {}!', x, k)
                ipl.filter_values(x, type='eq', setto=k, keys=targetnames[0])

    return ipl