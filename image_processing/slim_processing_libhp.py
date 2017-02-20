
import numpy as np
import vigra
import processing_lib as lib
from hdf5_slim_processing import Hdf5Processing as hp
from hdf5_slim_processing import RecursiveDict as Rdict
from concurrent import futures
import random
from simple_logger import SimpleLogger


def remove_small_objects(image, size_thresh,
                         parallelize=False, max_threads=5,
                         logger=SimpleLogger()):

    # Get the unique values of the segmentation including counts
    uniq, counts = np.unique(image, return_counts=True)

    # Keep all uniques that have a count smaller than size_thresh
    small_objs = uniq[counts < size_thresh]
    logger.logging('len(small_objs) == {}', len(small_objs))
    large_objs = uniq[counts >= size_thresh]
    logger.logging('len(large_objs) == {}', len(large_objs))

    if parallelize:

        if len(small_objs) > len(large_objs):
            def get_mask(image, lbl):
                return np.logical_not(image == lbl)

            with futures.ThreadPoolExecutor(max_threads) as do_stuff:
                tasks = [do_stuff.submit(get_mask, image, x) for x in large_objs]
            mask = np.all([x.result() for x in tasks], axis=0)

        else:

            def get_mask(image, lbl):
                return image == lbl

            with futures.ThreadPoolExecutor(max_threads) as do_stuff:
                tasks = [do_stuff.submit(get_mask, image, x) for x in large_objs]
            mask = np.any([x.result() for x in tasks], axis=0)

        timage = np.array(image)
        print mask.shape
        timage[mask] = 0

    else:

        if len(small_objs) > len(large_objs):

            timage = np.zeros(image.shape, dtype=image.dtype)
            for lo in large_objs:
                timage[image == lo] = lo

        else:

            timage = np.array(image)
            for so in small_objs:
                timage[timage == so] = 0

        # if len(small_objs) > len(large_objs):
        #     mask = np.logical_not(np.any([image == x for x in large_objs], axis=0))
        # else:
        #     mask = np.any([image == x for x in small_objs], axis=0)
        #
        # timage = np.array(image)
        # print mask.shape
        # timage[mask] = 0

    return timage


def remove_small_objects_relabel(
        image, size_thresh, relabel=True, consecutive_labels=True,
        parallelize=False, max_threads=5,
        logger=SimpleLogger()
):

    # Make sure all objects have their individual label
    if relabel:
        image = vigra.analysis.labelVolumeWithBackground(
            image.astype(np.uint32), neighborhood=6, background_value=0
        )

    # Remove objects smaller than size_thresh
    image = remove_small_objects(
        image, size_thresh, parallelize=parallelize, max_threads=max_threads,
        logger=logger
    )

    # Relabel the image for consecutive labels
    if consecutive_labels:
        vigra.analysis.relabelConsecutive(image, start_label=0, out=image)

    return image


def get_features(
        paths, shp, featureimages, featurelist, max_paths_per_label,
        logger=None, anisotropy=[1, 1, 1], return_pathlist=False,
        parallelized=False, max_threads=5
):
    """
    :param paths:
    :param featureimages:
    :param featurelist:
    :param max_paths_per_label:
    :param ipl:
    :param anisotropy:
    :param return_pathlist: When True a list of the path keys is returned in the same order as
        their features are stored -> Can be used for back-translation of the path classification
        to the respective object the path is in.
        It is basically a concatenation of the key list as yielded by the simultaneous iterator.
    :return:
    """

    newfeats = hp()

    # The path lengths only have to be computed once without using the vigra region features
    def compute_path_lengths(paths, anisotropy):

        path_lengths = []
        # for d, k, v, kl in paths.data_iterator():
        #     if type(v) is not type(paths):
        for path in paths:
            path_lengths.append(lib.compute_path_length(np.array(path), anisotropy))

        return np.array(path_lengths)

    # And only do it when desired
    pathlength = False
    try:
        featurelist.remove('Pathlength')
    except ValueError:
        # Means that 'Pathlength' was not in the list
        pass
    else:
        # 'Pathlength' was in the list and is now successfully removed
        pathlength = True

    if max_paths_per_label is not None:
        keylist = range(0, max_paths_per_label - 1)
        keylist = [str(x) for x in keylist]
    else:
        keylist = None

    if return_pathlist:
        pathlist = []

    # Iterate over all paths, yielding a list of one path per label object until no paths are left
    for i, keys, vals in paths.simultaneous_iterator(
            max_count_per_item=max_paths_per_label,
            keylist=keylist):
        # i is the iteration number
        # keys are respective labels and ids of the paths
        # vals are the coordinates of the path positions

        if return_pathlist:
            pathlist += keys

        if logger is not None:
            logger.logging('Working in iteration = {}', i)
            logger.logging('Keys: {}', keys)

        if not keys:
            continue

        # Create a working image
        image = np.zeros(shp, dtype=np.uint32)
        # And fill it with one path per label object
        c = 1
        for curk, curv in (dict(zip(keys, vals))).iteritems():
            curv = np.array(curv)
            if pathlength:
                if not newfeats.inkeys(['Pathlength']):
                    newfeats['Pathlength'] = np.array([lib.compute_path_length(curv, anisotropy)])
                else:
                    newfeats['Pathlength'] = np.concatenate((
                        newfeats['Pathlength'], [lib.compute_path_length(curv, anisotropy)]))
            curv = lib.swapaxes(curv, 0, 1)
            lib.positions2value(image, curv, c)
            c += 1

        # TODO: If this loop iterated over the parameter list it would be more broadly applicable
        if not parallelized:
            for d, k, v, kl in featureimages.data_iterator():

                if type(v) is not hp:

                    # Extract the region features of the working image
                    newnewfeats = hp(
                        data=vigra.analysis.extractRegionFeatures(
                            np.array(v).astype(np.float32),
                            image, ignoreLabel=0,
                            features=featurelist
                        )
                    )
                    # Pick out the features that we asked for
                    newnewfeats = newnewfeats.subset(*featurelist)

                    # Done: Extract feature 'Count' manually due to anisotropy
                    # Append to the recently computed list of features
                    for nk, nv in newnewfeats.iteritems():
                        nv = nv[1:]
                        if newfeats.inkeys(kl + [nk]):
                            try:
                                newfeats[kl + [nk]] = np.concatenate((newfeats[kl + [nk]], nv))
                            except ValueError:
                                pass
                        else:
                            newfeats[kl + [nk]] = nv

        elif parallelized:

            def extract_region_features(feat, im, ignore_label, featlist):
                return hp(vigra.analysis.extractRegionFeatures(
                            feat, im, ignoreLabel=ignore_label,
                            features=featlist
                ))


            logger.logging('Starting thread pool with a max of {} threads', max_threads)
            with futures.ThreadPoolExecutor(max_threads) as do_stuff:

                keys = []
                vals = []
                tasks = Rdict()

                for d, k, v, kl in featureimages.data_iterator(leaves_only=True):

                    # tasks[kl] = do_stuff.submit(
                    #     hp(vigra.analysis.extractRegionFeatures(
                    #         np.array(v).astype(np.float32), image, ignoreLabel=0,
                    #         features=featurelist
                    #     ))
                    # )
                    tasks[kl] = do_stuff.submit(
                        extract_region_features,
                        np.array(v).astype(np.float32), image, 0, featurelist
                    )
                    keys.append(kl)

            for kl in keys:

                newnewfeats = tasks[kl].result()
                newnewfeats = newnewfeats.subset(*featurelist)
                for nk, nv in newnewfeats.iteritems():
                    nv = nv[1:]
                    if newfeats.inkeys(kl + [nk]):
                        try:
                            newfeats[kl + [nk]] = np.concatenate((newfeats[kl + [nk]], nv))
                        except ValueError:
                            pass
                    else:
                        newfeats[kl + [nk]] = nv

    if return_pathlist:
        return newfeats, pathlist
    else:
        return newfeats


def compute_paths_for_class(
        indata, labelskey, pathendkey, disttransfkey, gtkey,
        params, for_class=True, ignore=[], logger=None, debug=False
):

    def shortest_paths(
            penaltypower, bounds,
            lbl, keylist_lblim,
            gt, disttransf, pathends,
            for_class=True, correspondence={},
            avoid_duplicates=True,
            max_paths_per_object=[], max_paths_per_object_seed=[], yield_in_bounds=False,
            return_pathim=True, minimum_alternative_label_count=0,
            logger=None
    ):
        """
        :param penaltypower:
        :param bounds:
        :param lbl:
        :param keylist_lblim: Needed for correspondence table
        :param disttransf:
        :param pathends:
        :param for_class:
            True: paths are computed for when endpoints are in the same ground truth oject
            False: paths are computed for when endpoints are in different ground truth objects
        :param correspondence:
        :param avoid_duplicates:
        :param max_paths_per_object:
        :param max_paths_per_object_seed:
        :param yield_in_bounds:
        :param return_pathim:
        :param minimum_alternative_label_count: Paths of merges (for_class=False) are removed if
            too little pixels of the merged object are found
        :param logger:
        :return:
        """

        # Pick up some statistics along the way
        stats_excluded_paths = 0
        statistics = Rdict()

        # Determine the endpoints of the current object
        indices = np.where(pathends)
        coords = zip(indices[0], indices[1], indices[2])

        # Make pairwise list of coordinates serving as source and target
        # First determine all pairings
        all_pairs = []
        for i in xrange(0, len(coords) - 1):
            for j in xrange(i + 1, len(coords)):
                all_pairs.append((coords[i], coords[j]))
        # And only use those that satisfy certain criteria:
        # a) Are in either the same gt object (for_class=True)
        #    or in different gt objects (for_class=False)
        # b) Are not in the correspondence list
        pairs = []
        label_pairs = []
        # if avoid_duplicates:
        new_correspondence = {}
        for pair in all_pairs:
            # Determine whether the endpoints are in different gt objects
            if (gt[pair[0]] == gt[pair[1]]) == for_class:
                # Check correspondence list if pairings were already computed in different image
                labelpair = tuple(sorted([gt[pair[0]], gt[pair[1]]]))
                if avoid_duplicates:
                    if labelpair not in correspondence.keys():
                        pairs.append(pair)
                        label_pairs.append(labelpair)
                        # new_correspondence[labelpair] = [keylist_lblim, lbl]
                        if logger is not None:
                            logger.logging('Found pairing: {}', labelpair)
                    else:
                        if logger is not None:
                            logger.logging('Pairing already in correspondence table: {}', labelpair)
                else:
                    pairs.append(pair)
                    if logger is not None:
                        logger.logging('Found pairing: {}', labelpair)
        # if avoid_duplicates:
        #     correspondence.update(new_correspondence)

        # Select a certain number of pairs if number is too high
        if max_paths_per_object:
            if len(pairs) > max_paths_per_object:
                if logger is not None:
                    logger.logging('Reducing number of pairs to {}', max_paths_per_object)
                if max_paths_per_object_seed:
                    random.seed(max_paths_per_object_seed)
                else:
                    random.seed()
                pairs = random.sample(pairs, max_paths_per_object)
                if logger is not None:
                    logger.logging('Modified pairs list: {}', pairs)

        # If pairs are found that satisfy all conditions
        if pairs:

            if logger is not None:
                logger.logging('Found {} pairings which satisfy all criteria', len(pairs))
            else:
                print 'Found {} pairings which satisfy all criteria'.format(len(pairs))

            # Pre-processing of the distance transform
            # a) Invert: the lowest values (i.e. the lowest penalty for the shortest path
            #    detection) should be at the center of the current process
            disttransf = lib.invert_image(disttransf)
            #
            # b) Set all values outside the process to infinity
            disttransf = lib.filter_values(disttransf, np.amax(disttransf), type='eq', setto=np.inf)
            #
            # c) Increase the value difference between pixels near the boundaries and pixels
            #    central within the processes. This increases the likelihood of the paths to
            #    follow the center of processes, thus avoiding short-cuts
            disttransf = lib.power(disttransf, penaltypower)

            # Compute the shortest paths according to the pairs list
            ps_computed, ps_in_bounds = lib.shortest_paths(
                disttransf, pairs, bounds=bounds, logger=logger,
                return_pathim=return_pathim, yield_in_bounds=yield_in_bounds
            )

            # Criteria for keeping paths which can only be computed after path computation
            if for_class:
                # A path without merge must not switch labels on the way!
                ps = []
                for i in xrange(0, len(ps_computed)):
                    if len(np.unique(
                            gt[ps_in_bounds[i][:, 0], ps_in_bounds[i][:, 1], ps_in_bounds[i][:, 2]])) == 1:
                        ps.append(ps_computed[i])
                        if logger is not None:
                            logger.logging('Path label = True')

                        # Add entry to correspondence table
                        if avoid_duplicates:
                            new_correspondence[label_pairs[i]] = [keylist_lblim, lbl]

                    else:
                        # The path switched objects multiple times on the way and is not added to the list\
                        if logger is not None:
                            logger.logging(
                                'Path starting and ending in label = {} had multiple labels and was excluded',
                                gt[tuple(ps_in_bounds[i][0])]
                            )

                        stats_excluded_paths += 1
            else:
                ps = []
                for i in xrange(0, len(ps_computed)):
                    un, counts = np.unique(
                        gt[ps_in_bounds[i][:, 0], ps_in_bounds[i][:, 1], ps_in_bounds[i][:, 2]],
                        return_counts=True
                    )
                    # At least two of the entries in counts have to be larger than the threshold
                    c = 0
                    for count in counts:
                        if count >= minimum_alternative_label_count:
                            c += 1
                        if c > 1:
                            break
                    if c > 1:
                        ps.append(ps_computed[i])

                        # Add entry to correspondence table
                        if avoid_duplicates:
                            new_correspondence[label_pairs[i]] = [keylist_lblim, lbl]

                    else:
                        if logger is not None:
                            logger.logging(
                                'Path starting in label {} and ending in {} only crossed one of the labels for {} voxels',
                                gt[tuple(ps_in_bounds[i][0])],
                                gt[tuple(ps_in_bounds[i][-1])],
                                np.min(counts)
                            )

            statistics['excluded_paths'] = stats_excluded_paths
            statistics['kept_paths'] = len(ps)
            return ps, new_correspondence, statistics

        else:
            statistics['excluded_paths'] = 0
            statistics['kept_paths'] = 0
            return [], new_correspondence, statistics

    def shortest_paths_wrapper(
            labelim, gt_im, dt_im, bc_im,
            lbl, kl, k,
            params,
            for_class=True,
            correspondence={},
            logger=None
    ):

        print 'Wrapper called...'

        # Create an image that contains only the one object
        lblim = np.zeros(labelim.shape, dtype=np.uint16)
        lblim[labelim == lbl] = lbl

        # Get the region of the one object
        bounds = lib.find_bounding_rect(lblim, s_=True)

        # Crop the label image
        lblim = lib.crop_bounding_rect(lblim, bounds)

        # Crop the gt as well
        gt_im = lib.crop_bounding_rect(gt_im, bounds=bounds)
        # Crop and mask the distance transform
        dt_im = lib.crop_bounding_rect(dt_im, bounds=bounds)
        dt_im[lblim == 0] = 0
        # Crop and mask border contacts
        bc_im = lib.crop_bounding_rect(bc_im, bounds=bounds)
        bc_im[lblim == 0] = 0
        # Done: Check for correctness

        # Compute all paths within this object which start and end in different
        #      gt-objects
        # Supply the correspondence table to this function and only compute a path
        #     if the respective correspondence is not found
        return shortest_paths(
            params['penaltypower'], bounds,
            lbl, kl + [k],
            gt_im, dt_im, bc_im,
            for_class=for_class, correspondence=correspondence,
            avoid_duplicates=params['avoid_duplicates'],
            max_paths_per_object=params['max_paths_per_object'],
            max_paths_per_object_seed=params['max_paths_per_object_seed'],
            yield_in_bounds=True,
            return_pathim=False,
            minimum_alternative_label_count=params['minimum_alternative_label_count'],
            logger=logger
        )

    correspondence_table = {}
    # correspondence_table (type=dict) should have the form:
    # {tuple(labels_in_gt_i): [kl_labelsimage_i, label_i]

    paths = hp()
    statistics = Rdict()

    if params['order_of_betas'] is not None:

        key_lists = []
        for i in params['order_of_betas']:
            key_lists += indata[labelskey].find_key_lists(i)

    else:

        key_lists = []
        for d, k, v, kl in indata[labelskey].data_iterator(leaves_only=True):
            key_lists.append(kl)

    # Iterate over segmentations
    # for d, k, v, kl in indata[labelskey].data_iterator(leaves_only=True, yield_short_kl=True):
    for i in key_lists:

        k = i[-1]
        kl = i[0:-1]

        if logger is not None:
            logger.logging('====================')
            logger.logging('Working on image {}', k)
            logger.logging('correspondence_table = {}', correspondence_table)
        else:
            print '===================='
            print 'Working on image {}'.format(k)
            print 'correspondence_table = {}'.format(correspondence_table)

        # Load the current segmentation image
        labelim = np.array(indata[labelskey][kl][k])
        # indata[labelskey][kl].populate(k)

        # TODO: Parallelize here
        # Iterate over all labels of that image (including cropping for speed-up)

        # Determine a list of present labels
        label_list = np.unique(labelim)
        label_list = filter(lambda x: x != 0, label_list)

        if params['parallelize']:

            logger.logging('Starting thread pool with {} threads', params['max_threads'])
            with futures.ThreadPoolExecutor(params['max_threads']) as do_stuff:
                tasks = Rdict()

                for lbl in label_list:

                    tasks[lbl] = do_stuff.submit(
                        shortest_paths_wrapper,
                        labelim,
                        np.array(indata[gtkey][kl][indata[gtkey][kl].keys()[0]]),
                        np.array(indata[disttransfkey][kl][k]['disttransf', 'raw']),
                        np.array(indata[pathendkey][kl][k]['contacts']),
                        lbl, kl, k,
                        params,
                        for_class=for_class,
                        correspondence=correspondence_table,
                        logger=logger
                    )

            for lbl in label_list:

                newpaths, new_correspondence_table, new_statistics = tasks[lbl].result()

                correspondence_table.update(new_correspondence_table)

                statistics[kl + [k] + [lbl]] = new_statistics

                # If new paths were detected
                if newpaths:
                    # Store them
                    # paths.merge(newpaths)

                    pskeys = range(0, len(newpaths))
                    paths[kl + [k] + [lbl]] = hp(data=dict(zip(pskeys, newpaths)))

                    if logger is not None:
                        logger.logging(
                            'Found {} paths in image {} at label {}', len(newpaths), k, lbl
                        )
                        logger.logging('-------------------')
                    else:
                        print 'Found {} paths in image {} at label {}'.format(len(newpaths), k, lbl)
                        print '-------------------'

        else:

            # Iterate over these labels
            for lbl in label_list:

                newpaths, new_correspondence_table, new_statistics = shortest_paths_wrapper(
                    labelim,
                    np.array(indata[gtkey][kl][indata[gtkey][kl].keys()[0]]),
                    np.array(indata[disttransfkey][kl][k]['disttransf', 'raw']),
                    np.array(indata[pathendkey][kl][k]['contacts']),
                    lbl, kl, k,
                    params,
                    for_class=for_class,
                    correspondence=correspondence_table,
                    logger=logger
                )

                correspondence_table.update(new_correspondence_table)

                statistics[kl + [k] + [lbl]] = new_statistics

                # If new paths were detected
                if newpaths:
                    # Store them
                    # paths.merge(newpaths)

                    pskeys = range(0, len(newpaths))
                    paths[kl + [k] + [lbl]] = hp(data=dict(zip(pskeys, newpaths)))

                    if logger is not None:
                        logger.logging(
                            'Found {} paths in image {} at label {}', len(newpaths), k, lbl
                        )
                        logger.logging('-------------------')
                    else:
                        print 'Found {} paths in image {} at label {}'.format(len(newpaths), k, lbl)
                        print '-------------------'

        # # Unload the current segmentation image
        # indata[labelskey][kl].unpopulate()

    return paths, statistics


def rf_combine_sources(features, pathlist):
    """
    features:
    ---------

    What we have:
    [somesource_0]
        [features]: [f_00, f_10, ..., f_n0]    # with n being the number of paths
    ...
    [somesource_N]:
        [features]: [f_0N, f_1N, ..., f_nN]

    What we want to have:
    [features]: [f_00, ..., f_n0, f_01, ..., f_n1, ..., f_0N, ..., fnN]

    pathlist:
    ---------

    What we have:
    [somesource_0]: [kl_00, kl_10, ..., kl_n0]    # with n being the number of paths
    ...
    [somesource_N]: [kl_0N, kl_1N, ..., kl_nN]

    What we want to have:
    [somesource_0 + kl_00, ..., somesource_N + kl_nN]

    :return:
    """

    outfeatures = hp()
    newpathlist = []

    # print 'Starting rf_combine_sources_new\n'

    for d, k, v, kl in pathlist.data_iterator(leaves_only=True):
        # print kl

        newpathlist += [kl + list(x) for x in pathlist[kl]]

        for d2, k2, v2, kl2 in features[kl].data_iterator(leaves_only=True):
            if outfeatures.inkeys(kl2):
                outfeatures[kl2] \
                    = np.concatenate((outfeatures[kl2], np.array(v2)), axis=0)
            else:
                outfeatures[kl2] = np.array(v2)

    return outfeatures, newpathlist


def rf_make_feature_array_with_keylist(features, keylist):

    featsarray = None

    for k in keylist:
        if featsarray is None:
            if features[k].ndim == 1:
                featsarray = np.array([features[k]])
            elif features[k].ndim == 2:
                featsarray = np.array(features[k]).swapaxes(0, 1)
        else:
            if features[k].ndim == 1:
                featsarray = np.concatenate((featsarray, [features[k]]), 0)
            elif features[k].ndim == 2:
                featsarray = np.concatenate((featsarray, features[k].swapaxes(0, 1)))

    featsarray = featsarray.swapaxes(0, 1)

    return featsarray


from sklearn.ensemble import RandomForestClassifier as Skrf
def random_forest(trainfeatures, testfeatures, debug=False, balance=False, logger=None):

    # print '\n---\n'
    # print 'trainfeatures: '
    # print trainfeatures
    # print '\n---\n'
    # print 'testfeatures'
    # print testfeatures
    # print '\n---\n'

    if balance:
        lentrue = len(trainfeatures['true'])
        lenfalse = len(trainfeatures['false'])
        if lentrue > lenfalse:
            trainfeatures['true'] = np.array(random.sample(trainfeatures['true'].tolist(), lenfalse))
        else:
            trainfeatures['false'] = np.array(random.sample(trainfeatures['false'].tolist(), lentrue))

    traindata, trainlabels = rf_make_forest_input(trainfeatures)
    testdata, testlabels = rf_make_forest_input(testfeatures)
    if logger is None:
        print traindata.shape
        print testdata.shape
        print trainlabels.shape
    else:
        logger.logging('Data sizes using balance = {}:', balance)
        logger.logging('    traindata.shape = {}', traindata.shape)
        logger.logging('    testdata.shape = {}', testdata.shape)
        logger.logging('    trainlabels.shape = {}', trainlabels.shape)

    if debug:
        # Check if any item from traindata also occurs in testdata
        c = 0
        for i in traindata.tolist():
            if i in testdata.tolist():
                c += 1
        print '{} items were identical.'.format(c)

    rf = Skrf()
    rf.fit(traindata, trainlabels)

    result = rf.predict(testdata)

    # print result.shape
    # print testlabels.shape

    return zip(result, testlabels)


def rf_make_forest_input(features):

    lentrue = features['true'].shape[0]
    lenfalse = features['false'].shape[0]

    classes = np.concatenate((np.ones((lentrue,)), np.zeros((lenfalse,))))

    data = np.concatenate((features['true'], features['false']), axis=0)

    data = rf_eliminate_invalid_entries(data)

    return [data, classes]


def rf_eliminate_invalid_entries(data):

    data[np.isnan(data)] = 0
    data[np.isinf(data)] = 0

    data = data.astype(np.float32)

    return data


from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
def new_eval(result, truth):

    return {
        'precision': precision_score(truth, result, average='binary', pos_label=0),
        'recall': recall_score(truth, result, average='binary', pos_label=0),
        'f1': f1_score(truth, result, average='binary', pos_label=0),
        'accuracy': accuracy_score(truth, result)
    }