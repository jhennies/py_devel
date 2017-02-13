
import numpy as np
import vigra
import processing_lib as lib
from hdf5_slim_processing import Hdf5Processing as hp
from hdf5_slim_processing import RecursiveDict as Rdict
from concurrent import futures

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

    keylist = range(0, max_paths_per_label - 1)
    keylist = [str(x) for x in keylist]

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
        # TODO: Parallelize here!

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
