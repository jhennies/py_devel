
import os
import inspect
# from hdf5_image_processing import Hdf5ImageProcessing as IP, Hdf5ImageProcessingLib as ipl
from hdf5_slim_processing import Hdf5Processing as hp
# from hdf5_processing import RecursiveDict as rdict
from hdf5_slim_processing import RecursiveDict as Rdict
from shutil import copy, copyfile
import numpy as np
import matplotlib.pyplot as plt
import slim_processing_libhp as libhp
import sys
from yaml_parameters import YamlParams
import pickle


__author__ = 'jhennies'


def load_images(filepath, skeys=None, recursive_search=False, logger=None):

    if logger is not None:
        logger.logging('Loading data from \n{}', filepath)
    else:
        print 'Loading data from \n{}'.format(filepath)

    data = hp()

    data.data_from_file(
        filepath=filepath,
        skeys=skeys,
        recursive_search=recursive_search,
        nodata=True
    )

    return data


def features_of_paths(yparams):

    all_params = yparams.get_params()

    # Zero'th layer:
    # --------------
    zeroth = Rdict(all_params['features_of_paths'])
    if 'default' in zeroth:
        zeroth_defaults = zeroth.pop('default')
    else:
        zeroth_defaults = hp()

    pathlist = hp()
    pathlistfile = zeroth_defaults['targets', 'pathlist']
    pathlistfile = all_params[pathlistfile[0]] + all_params[pathlistfile[1]]

    for exp_lbl, experiment in zeroth.iteritems():

        # First layer
        # -----------
        # An experiment is now selected and performed
        yparams.logging('\n\nPerforming experiment {}\n==============================', exp_lbl)

        final = zeroth_defaults.dcp()
        final.merge(experiment)

        exp_sources = final['sources']
        exp_params = final['params']
        exp_targets = final['targets']

        def val(x):
            return x
        yparams.logging('exp_sources = \n{}', exp_sources.datastructure2string(function=val))
        yparams.logging('exp_params = \n{}', exp_sources.datastructure2string(function=val))
        yparams.logging('exp_targets = \n{}', exp_targets.datastructure2string(function=val))

        # Load feature images
        # -------------------
        featureims = hp()
        for k, v in exp_sources['featureims'].iteritems():
            skeys = None
            if 'skeys' in v[2]:
                skeys = v[2]['skeys']
            featureims[k] = load_images(
                all_params[v[0]] + all_params[v[1]],
                skeys=skeys, logger=yparams
            )
        yparams.logging(
            '\nFeatureims datastructure: \n\n{}',
            featureims.datastructure2string(maxdepth=4)
        )

        for exp_class_lbl, exp_class_src in exp_sources['paths'].iteritems():

            yparams.logging('\nWorking on {}\n------------------------------', exp_class_lbl)

            # Load paths
            # ----------
            skeys = None
            if 'skeys' in exp_class_src[2]:
                skeys = exp_class_src[2]['skeys']
            paths = load_images(
                all_params[exp_class_src[0]] + all_params[exp_class_src[1]],
                skeys=skeys, logger=yparams
            )
            yparams.logging(
                '\nPaths datastructure: \n\n{}',
                paths.datastructure2string(maxdepth=4)
            )

            # Iterate over the paths
            for d, k, v, kl in paths[exp_class_src[2]['skeys'][0]].data_iterator(
                    leaves_only=True, yield_short_kl=True, maxdepth=3
            ):
                yparams.logging('\nPath keylist: {}\n..............................', kl + [k])

                segm_kl = kl + [k]
                imgs_kl = kl
                yparams.logging('segm_kl = {}', segm_kl)
                yparams.logging('imgs_kl = {}', imgs_kl)

                # Bild an input featureims dict for the path computation
                infeatims = hp()
                sourcelist = exp_sources['featureims'].dcp()
                if 'segmentation' in sourcelist:
                    infeatims['segmentation'] = featureims['segmentation'][segm_kl]
                    sourcelist.pop('segmentation')
                for source in sourcelist:
                    # TODO: This is not nice... Here I try to remove a redundant key
                    infeatims[source] = featureims[source][imgs_kl][featureims[source][imgs_kl].keys()[0]]
                # infeatims.populate()

                # Bild an input dict for true paths
                inpaths = v.dcp()
                # inpaths.populate()

                # Get the necessary image shape
                for d2, k2, v2, kl2 in infeatims.data_iterator(leaves_only=True):
                    im_shp = v2.shape
                    break

                features = hp()
                # import time
                # start = time.time()
                # print 'Starting get_features'
                features[exp_lbl][[exp_class_lbl] + kl + [k]], pathlist[exp_lbl][[exp_class_lbl] + kl + [k]] = libhp.get_features(
                    inpaths, np.array(im_shp)[0:3],
                    infeatims, list(exp_params['features']),
                    exp_params['max_paths_per_label'], logger=yparams,
                    anisotropy=exp_params['anisotropy'], return_pathlist=True,
                    parallelized=exp_params['parallelize'], max_threads=exp_params['max_threads']
                )
                # print 'Stopping get_features'
                # stop = time.time()
                # print stop-start

                yparams.logging(
                    '\nFeatures datastructure: \n\n{}',
                    features.datastructure2string(maxdepth=4)
                )

                # Write the result to file
                features.write(
                    filepath=all_params[exp_targets['features'][0]] + all_params[exp_targets['features'][1]]
                )

    with open(pathlistfile, 'wb') as f:
        pickle.dump(pathlist, f)


def run_features_of_paths(yamlfile, logging=True):

    yparams = YamlParams(filename=yamlfile)
    params = yparams.get_params()

    # Logger stuff
    yparams.set_indent(1)
    yparams.startlogger(
        filename=params['resultfolder'] + 'features_of_paths.log',
        type='w', name='FeaturesOfPaths'
    )

    try:

        features_of_paths(yparams)

        yparams.logging('')
        yparams.stoplogger()

    except:

        yparams.errout('Unexpected error')


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters_ref.yml'

    run_features_of_paths(yamlfile, logging=False)