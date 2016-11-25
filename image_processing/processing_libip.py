
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
    
    
def compute_faces(ipl, key):

    faces = IPL(data={key: ipl[key]})
    shp = faces[key].shape
    ipl.logging('Computing faces ...')
    faces.get_faces_with_neighbors(keys=key, rtrntype=IPL)

    # startpoints = ipl['faces', keys[0]].keys()
    additionalinfo = IPL()
    startpoints = IPL(data={'xyf': shp[2],
                   'xyb': shp[2],
                   'xzf': shp[1],
                   'xzb': shp[1],
                   'yzf': shp[0],
                   'yzb': shp[0]})
    areas = IPL(data={'xyf': np.s_[shp[2]:shp[2] + shp[0], shp[2]:shp[2] + shp[1]],
             'xyb': np.s_[shp[2]:shp[2] + shp[0], shp[2]:shp[2] + shp[1]],
             'xzf': np.s_[shp[1]:shp[1] + shp[0], shp[1]:shp[1] + shp[2]],
             'xzb': np.s_[shp[1]:shp[1] + shp[0], shp[1]:shp[1] + shp[2]],
             'yzf': np.s_[shp[0]:shp[0] + shp[1], shp[0]:shp[0] + shp[2]],
             'yzb': np.s_[shp[0]:shp[0] + shp[1], shp[0]:shp[0] + shp[2]]})

    additionalinfo[key, 'startpoints'] = startpoints
    additionalinfo[key, 'areas'] = areas

    return (faces, additionalinfo)
    
    
def find_border_centroids(ipl, faces, key, facesinfo, facesd, resultkey, resultshp):
    """
    :param ipl: the result is stored here using resultkey
    :param faces: ipl containing faces as returned by compute_faces
    :param key: key of the image in ipl
    :param facesinfo: ipl containing facesinfo as returned by compute_faces
    :param facesd: ipl containing the faces of the distance transform as returned by compute_faces
    :param resultkey:
    :return:
    """

    ipl[resultkey, key] = np.zeros(resultshp)

    for k, startpoint in facesinfo[key, 'startpoints'].iteritems():

        # bounds = (shp[0],) * 2
        for lbl, lblim in faces[key].label_image_iterator(key=k, background=0, area=facesinfo[key, 'areas', k]):

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
                curdist = np.array(facesd[key, k])
                curdist[curobj == False] = 0

                # Detect the global maximum of this object
                amax = np.amax(curdist)
                curdist[curdist < amax] = 0
                curdist[curdist > 0] = lbl
                # Only one pixel is allowed to be selected
                bds = lib.find_bounding_rect(curdist)
                centroid = (int((bds[1][0] + bds[1][1]-1) / 2), int((bds[0][0] + bds[0][1]-1) / 2))

                # Now translate the calculated centroid to the position within the orignial 3D volume
                centroidm = (centroid[0] - startpoint, centroid[1] - startpoint)
                # ipl.logging('centroidxy = {}', centroidm)
                # Set the pixel
                try:
                    if centroidm[0] < 0 or centroidm[1] < 0:
                        raise IndexError
                    else:
                        if k == 'xyf':
                            ipl[resultkey, key][centroidm[0], centroidm[1], 0] = lbl
                        elif k == 'xyb':
                            ipl[resultkey, key][centroidm[0], centroidm[1], -1] = lbl
                        elif k == 'xzf':
                            ipl[resultkey, key][centroidm[0], 0, centroidm[1]] = lbl
                        elif k == 'xzb':
                            ipl[resultkey, key][centroidm[0], -1, centroidm[1]] = lbl
                        elif k == 'yzf':
                            ipl[resultkey, key][0, centroidm[0], centroidm[1]] = lbl
                        elif k == 'yzb':
                            ipl[resultkey, key][-1, centroidm[0], centroidm[1]] = lbl
                except IndexError:
                    pass


def find_orphans(ipl, bordercontacts, key, tkey):

    non_orphan_labels = []
    for k, v in ipl['faces', key].iteritems():
        non_orphan_labels = np.unique(np.append(v, non_orphan_labels))

    all_labels = np.unique(ipl[key])
    orphan_labels = list(set(all_labels).difference(non_orphan_labels))

    if orphan_labels:
        bordercontacts[tkey] = ipl.getlabel(orphan_labels, keys=key, return_only=True)


def find_border_contacts(ipl, keys, tkey, debug=False):
    # Each key in keys needs a corresponding entry in 'disttransf'.
    # Example for keys=('labels1', 'labels2')
    # ipl:
    #   'labels1': labelimage
    #   'labels2': labelimage
    #   'disttransf':
    #       'labels1': distance transform of 'labels1'
    #       'labels2': distance transform of 'labels2'

    for key in keys:

        ipl.logging('Finding border contacts for key = {}', key)

        ipl.populate(key)
        ipl['disttransf'].populate(key)

        # For each of the 6 faces compute the objects which are touching it and the corresponding local maxima of the
        # distance transform
        faces, faces_info = compute_faces(ipl, key)
        facesd, facesd_info = compute_faces(ipl['disttransf'], key)

        find_border_centroids(ipl, faces, key, faces_info, facesd, tkey, ipl[key].shape)

        if debug:
            ipl[tkey, 'overlay_{}'.format(key)] = np.array([
                (ipl[tkey, key] > 0).astype(np.float32),
                (ipl[key].astype(np.float32)/np.amax(ipl[key])).astype(np.float32),
                (ipl['disttransf', key]/np.amax(ipl['disttransf', key])).astype(np.float32)
            ])


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
        if seed:
            random.seed(seed)
        else:
            random.seed()

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


def find_shortest_path(ipl, penaltypower, bounds, disttransf, locmax,
                       max_end_count=[], max_end_count_seed=[]):

    # Modify distancetransform
    #
    # a) Invert: the lowest values (i.e. the lowest penalty for the shortest path detection) should be at the center of
    #    the current process
    disttransf = lib.invert_image(disttransf)
    #
    # b) Set all values outside the process to infinity
    disttransf = lib.filter_values(disttransf, np.amax(disttransf), type='eq', setto=np.inf)
    #
    # c) Increase the value difference between pixels near the boundaries and pixels central within the processes
    #    This increases the likelihood of the paths to follow the center of processes, thus avoiding short-cuts
    disttransf = lib.power(disttransf, penaltypower)

    # Get local maxima
    indices = np.where(locmax)
    coords = zip(indices[0], indices[1], indices[2])
    ipl.logging('Local maxima coordinates: {}', coords)

    # Select a certain number of path end points
    if max_end_count:
        if len(coords) > max_end_count:
            ipl.logging('Reducing number of coordinates to {}', max_end_count)
            if max_end_count_seed:
                random.seed(max_end_count_seed)
            else:
                random.seed()
            coords = random.sample(coords, max_end_count)
            ipl.logging('Modified coordinate list: {}', coords)

    # Make pairwise list of coordinates that will serve as source and target
    pairs = []
    for i in xrange(0, len(coords)-1):
        for j in xrange(i+1, len(coords)):
            pairs.append((coords[i], coords[j]))

    paths, pathim = lib.shortest_paths(disttransf, pairs, bounds=bounds, hfp=ipl)

    # # Make sure no empty paths lists are returned
    # paths = [x for x in paths if x.any()]
    return paths, pathim


def paths_of_labels(
        ipl, labelkey, pathendkey, disttransfkey, thisparams, ignore=[],
        max_end_count=[], max_end_count_seed=[]
):

    # if type(locmaxkeys) is str:
    #     locmaxkeys = (locmaxkeys,)

    # This results in the format:
    # more_keys = (locmaxkeys[0], ..., locmaxkeys[n], disttransfkey)
    more_keys = (pathendkey, disttransfkey)

    paths = IPL()
    # for k in locmaxkeys:
    paths['pathsim'] = np.zeros(ipl[labelkey].shape)

    for lbl, lblim, more_ims, bounds in ipl.label_image_bounds_iterator(
        key=labelkey, background=0, more_keys=more_keys,
        maskvalue=0, value=0
    ):
        if lbl in ignore:
            continue
        # The format of more_ims is:
        # more_ims = {locmaxkeys[0]: locmax_1,
        #                ...
        #             locmaxkeys[n]: locmax_n,
        #             disttransfkey: disttransf}

        ipl.logging('======================\nWorking on label = {}', lbl)
        # ipl.logging('more_ims structure:\n---\n{}', more_ims.datastructure2string())
        ipl.logging('bounds = {}', bounds)

        if np.amax(more_ims[pathendkey]) > 0:
            ps, pathsim = find_shortest_path(
                ipl, thisparams['penaltypower'], bounds,
                more_ims[disttransfkey], more_ims[pathendkey],
                max_end_count=max_end_count,
                max_end_count_seed=max_end_count_seed
            )

            # Only store the path if the path-calculation successfully determined a path
            # Otherwise an empty list would be stored
            if ps:
                ipl.logging('Number of paths found: {}', len(ps))

                pskeys = range(0, len(ps))
                ps = IPL(data=dict(zip(pskeys, ps)))

                paths['path', lbl] = ps
                # paths['pathsim'] = pathsim

                paths['pathsim'][bounds][pathsim > 0] = pathsim[pathsim > 0]


    paths['overlay'] = np.array(
        [paths['pathsim'],
        ipl[labelkey].astype(np.float32) / np.amax(ipl[labelkey]),
        vigra.filters.multiBinaryDilation(ipl[pathendkey].astype(np.uint8), 5)]
    )

    return paths


def find_shortest_path_labelpairs(ipl, penaltypower, bounds, disttransf, locmax,
                       labels, labelgroup,
                       max_end_count=[], max_end_count_seed=[]):

    # Modify distancetransform
    #
    # a) Invert: the lowest values (i.e. the lowest penalty for the shortest path detection) should be at the center of
    #    the current process
    disttransf = lib.invert_image(disttransf)
    #
    # b) Set all values outside the process to infinity
    disttransf = lib.filter_values(disttransf, np.amax(disttransf), type='eq', setto=np.inf)
    #
    # c) Increase the value difference between pixels near the boundaries and pixels central within the processes
    #    This increases the likelihood of the paths to follow the center of processes, thus avoiding short-cuts
    disttransf = lib.power(disttransf, penaltypower)

    # The situation:
    # We have multiple objects (n > 1) of unknown number.
    # We want to extract the local maxima within each object individually and create a
    #   list of all possible partners (pairs)
    # Each partner of a pair has to be located within a different object (label area)
    #
    # Approach 1:
    #   For each locmax iterate over all other locmaxs and write pairs, which satisfy the
    #   desired condition, to the pairs list
    #
    # Approach 2:
    #   For each label object iterate over all the others and append all possible locmax
    #   pairs to the pairs list
    #   Probably faster than approach 1 when implemented correctly? Someone should test that...

    # Approach 2 in its implemented form
    pairs = []
    for i in xrange(0, len(labelgroup)-1):

        # Find the path endpoints within one label object
        indices_i = np.where((labels == labelgroup[i]) & (locmax > 0))
        indices_i = zip(indices_i[0], indices_i[1], indices_i[2])
        # Throw out some if the list is too long
        if max_end_count:
            if len(indices_i) > max_end_count:
                ipl.logging('Reducing number of coordinates in indices_i to {}', max_end_count)
                random.seed()
                indices_i = random.sample(indices_i, max_end_count)

        if indices_i:
            for j in xrange(i+1, len(labelgroup)):

                # Find the path endpoints within an merged label object
                indices_j = np.where((labels == labelgroup[j]) & (locmax > 0))
                indices_j = zip(indices_j[0], indices_j[1], indices_j[2])
                # Throw out some if the list is too long
                if max_end_count:
                    if len(indices_j) > max_end_count:
                        ipl.logging('Reducing number of coordinates in indices_j to {}', max_end_count)
                        random.seed()
                        indices_j = random.sample(indices_j, max_end_count)

                if indices_j:
                    ipl.logging('Ind_i = {}\nInd_j = {}', indices_i, indices_j)
                    # Now, lets do some magic!
                    pairs = pairs + zip(indices_i * len(indices_j), sorted(indices_j * len(indices_i)))

    paths, pathim = lib.shortest_paths(disttransf, pairs, bounds=bounds, hfp=ipl)

    return paths, pathim


def paths_of_labelpairs(
        ipl, lblsmkey, lblskey, changehashkey, pathendkey, disttransfkey, thisparams, ignore,
        max_end_count=[], max_end_count_seed=[]
):

    # This results in the format:
    # more_keys = (locmaxkeys[0], ..., locmaxkeys[n], disttransfkey, lblskey)
    more_keys = (pathendkey, disttransfkey, lblskey)

    paths = IPL()
    paths['pathsim'] = np.zeros(ipl[lblsmkey].shape)

    # Determine labellist from change_hash (keys are strings due to h5 saving)
    labellist = ipl[changehashkey].keys()
    labellist = [int(x) for x in labellist]

    for lbl, lblim, more_ims, bounds in ipl.label_image_bounds_iterator(
        key=lblsmkey, background=0, more_keys=more_keys,
        maskvalue=0, value=0, labellist=labellist
    ):
        if lbl in ignore:
            continue
        # The format of more_ims is:
        # more_ims = {locmaxkeys[0]: locmax_1,
        #                ...
        #             locmaxkeys[n]: locmax_n,
        #             disttransfkey: disttransf,
        #             lblskey: lables}

        ipl.logging('======================\nWorking on label = {}', lbl)
        # ipl.logging('more_ims structure:\n---\n{}', more_ims.datastructure2string())
        ipl.logging('bounds = {}', bounds)

        if np.amax(more_ims[pathendkey]) > 0:
            ps, pathsim = find_shortest_path_labelpairs(
                ipl, thisparams['penaltypower'], bounds,
                more_ims[disttransfkey],
                more_ims[pathendkey],
                more_ims[lblskey],
                ipl[changehashkey][str(lbl)],
                max_end_count=max_end_count,
                max_end_count_seed=max_end_count_seed
            )

            # Only store the path if the path-calculation successfully determined a path
            # Otherwise an empty list would be stored
            if ps:
                ipl.logging('Number of paths found: {}', len(ps))

                pskeys = range(0, len(ps))
                ps = IPL(data=dict(zip(pskeys, ps)))

                paths['path', lbl] = ps
                # paths[lblsmkey, locmaxkeys[i], 'pathsim'] = pathsim

                paths['pathsim'][bounds][pathsim > 0] = pathsim[pathsim > 0]

        paths['overlay'] = np.array([
            paths['pathsim'],
            ipl[lblskey].astype(np.float32) / np.amax(ipl[lblskey]),
            vigra.filters.multiBinaryDilation(ipl[pathendkey].astype(np.uint8), 5)
        ])

    return paths


def get_features(paths, featureimages, featurelist, max_paths_per_label, ipl=None):

    newfeats = IPL()

    keylist = range(0, max_paths_per_label)
    keylist = [str(x) for x in keylist]

    # Iterate over all paths, yielding a list of one path per label object until no paths are left
    for i, keys, vals in paths.simultaneous_iterator(
            max_count_per_item=max_paths_per_label,
            keylist=keylist):

        if ipl is not None:
            ipl.logging('Working in iteration = {}', i)
            ipl.logging('Keys: {}', keys)

        if not keys:
            continue

        # Create a working image
        image = np.zeros(np.array(featureimages.yield_an_item()).shape, dtype=np.uint32)
        # And fill it with one path per label object
        c = 0
        for curk, curv in (dict(zip(keys, vals))).iteritems():
            curv = np.array(curv)
            curv = lib.swapaxes(curv, 0, 1)
            lib.positions2value(image, curv, c)
            c += 1

        for d, k, v, kl in featureimages.data_iterator():

            if type(v) is not IPL:

                # Extract the region features of the working image
                # TODO: Extract feature 'Count' manually due to anisotropy
                newnewfeats = IPL(
                    data=vigra.analysis.extractRegionFeatures(
                        np.array(v).astype(np.float32),
                        image, ignoreLabel=0,
                        features=featurelist
                    )
                )
                # Append to the recently computed list of features
                for nk, nv in newnewfeats.iteritems():
                    nv = nv[1:]
                    if nk in newfeats:
                        try:
                            newfeats[kl + [nk]] = np.concatenate((newfeats[kl + [nk]], nv))
                        except ValueError:
                            pass
                    else:
                        newfeats[kl + [nk]] = nv

    return newfeats


def features_of_paths(ipl, paths_true, paths_false, featureims_true, featureims_false):
    # def features_of_paths(ipl, disttransf_images, feature_images, thisparams):
    """
    The following datastructure is necessary for the dicts 'disttransf_images' and 'feature_images':
    true
    .   [locmax_name]
    .   .   [feature_name]
    false
    .   [locmax_name]
    .   .   [feature_name]

    ipl has this datastructure:
    true
    .   [locmax_name]
    .   .   [labels]
    .   .   .   [paths]
    false
    .   [locmax_name]
    .   .   [labels]
    .   .   .   [paths]

    :param ipl:
    :param disttransf_images:
    :param feature_images:
    :param thisparams:
    :return:
    """

    params = ipl.get_params()
    thisparams = params['features_of_paths']

    features = IPL()

    features['true'] = get_features(
        paths_true, featureims_true,
        thisparams['features'],
        thisparams['max_paths_per_label'], ipl=ipl
    )

    features.write(
        filepath=params['intermedfolder'] + params['featuresfile'],
        keys='true', search=True
    )
    features['true'] = None

    features['false'] = get_features(
        paths_false, featureims_false,
        thisparams['features'],
        thisparams['max_paths_per_label'], ipl=ipl
    )

    features.write(
        filepath=params['intermedfolder'] + params['featuresfile'],
        keys='false', search=True
    )
