
import os
import inspect
from hdf5_image_processing import Hdf5ImageProcessing as IP, Hdf5ImageProcessingLib as ipl
from hdf5_processing import RecursiveDict as rdict
from shutil import copy, copyfile
import numpy as np
import matplotlib.pyplot as plt
import processing_libip as libip
import sys
from yaml_parameters import YamlParams


__author__ = 'jhennies'


def load_images(filepath, skeys=None, recursive_search=False, logger=None):

    if logger is not None:
        logger.logging('Loading data from \n{}', filepath)
        logger.logging('With skeys = {}', skeys)
    else:
        print 'Loading data from \n{}'.format(filepath)

    data = ipl()

    data.data_from_file(
        filepath=filepath,
        skeys=skeys,
        recursive_search=recursive_search,
        nodata=True
    )

    return data


def compute_paths(yparams):

    all_params = yparams.get_params()

    # Zero'th layer:
    # --------------
    zeroth = rdict(all_params['compute_paths'])
    if 'default' in zeroth:
        zeroth_defaults = zeroth.pop('default')
    else:
        zeroth_defaults = ipl()

    for exp_lbl, experiment in zeroth.iteritems():

        # First layer
        # -----------
        # An experiment is now selected and performed
        yparams.logging('Performing experiment {}\n==============================\n', exp_lbl)

        first = zeroth_defaults.dcp()
        first.merge(experiment)
        if 'default' in first:
            first_defaults = first.pop('default')
        else:
            first_defaults = ipl()

        for exp_class_lbl in ['truepaths', 'falsepaths']:

            # Final layer
            # -----------
            # The true or false paths for the current experiment are here computed, respectively
            yparams.logging('Computing {}...\n------------------------------\n', exp_class_lbl)
            final = first_defaults.dcp()
            final.merge(first[exp_class_lbl])

            exp_sources = final['sources']
            exp_params = final['params']
            exp_target = final['target']

            # Load the necessary images
            data=ipl()
            for datakey, content in exp_sources.iteritems():
                data[datakey] = load_images(
                    all_params[content[0]] + all_params[content[1]],
                    skeys=content[2]['skeys'],
                    recursive_search=False,
                    logger=yparams
                )

            yparams.logging('\nInitial datastructure: \n\n{}', data.datastructure2string(maxdepth=4))
            yparams.logging('experiment_params: \n{}', exp_params)

            # Compute the paths
            # -----------------
            paths = ipl()
            for_class = False
            if exp_class_lbl == 'truepaths':
                for_class = True
            paths[exp_lbl][exp_class_lbl] = libip.compute_paths_for_class(
                data, 'segm', 'conts', 'dt', 'gt',
                exp_params, for_class=for_class, ignore=[], debug=all_params['debug'],
                logger=yparams
            )

            yparams.logging(
                '\nPaths datastructure after running {}: \n\n{}',
                exp_class_lbl,
                paths.datastructure2string()
            )

            # Save the result to disk
            # -----------------------
            targetfile = all_params[exp_target[0]] + all_params[exp_target[1]]
            paths.write(filepath=targetfile)


    #
    # # Load and initialize general stuff (valid for all experiments)
    # # -------------------------------------------------------------
    #
    # params = yparams.get_params()
    # thisparams = rdict(params['compute_paths'])
    #
    # if 'general_params' in thisparams:
    #     general_params = rdict(thisparams['general_params'])
    # else:
    #     general_params = rdict()
    #
    # general_sources = ipl()
    # if 'general_sources' in thisparams['sources']:
    #     for datakey, content in thisparams['sources', 'general_sources'].iteritems():
    #         general_sources[datakey] = load_images(
    #             params[content[0]] + params[content[1]],
    #             skeys=content[2]['skeys'],
    #             recursive_search=False,
    #             logger=yparams
    #         )
    #
    # # Set targetfile
    # targetfile = params[thisparams['target'][0]] + params[thisparams['target'][1]]
    #
    # # Starting the experiments
    # # ------------------------
    #
    # for experiment, datakeys in thisparams['sources'].iteritems():
    #
    #     if experiment != 'general_sources':
    #
    #         yparams.logging('Performing experiment {}\n==============================\n', experiment)
    #
    #         # Load and initialize specific stuff for this experiment (may overwrite the
    #         # general stuff)
    #         # -------------------------------------------------------------------------
    #
    #         data = general_sources.dcp()
    #         experiment_params = general_params.dcp()
    #         for datakey, content in datakeys.iteritems():
    #
    #             if datakey != 'params':
    #
    #                 # Load the necessary images
    #                 data[datakey] = load_images(
    #                     params[content[0]] + params[content[1]],
    #                     skeys=content[2]['skeys'],
    #                     recursive_search=False,
    #                     logger=yparams
    #                 )
    #
    #             else:
    #
    #                 # Load parameters
    #                 experiment_params.merge(content)
    #
    #         yparams.logging('\nInitial datastructure: \n\n{}', data.datastructure2string(maxdepth=4))
    #         yparams.logging('experiment_params: \n{}', experiment_params)
    #
    #         # Compute the paths
    #         # -----------------
    #
    #         paths = ipl()
    #         paths[experiment]['truepaths'] = libip.compute_paths_for_class(
    #             data, 'segm_true', 'conts_true', 'dt_true', 'gt_true',
    #             experiment_params, for_class=True, ignore=[], debug=params['debug'],
    #             logger=yparams
    #         )
    #         paths.dss()
    #         paths[experiment]['falsepaths'] = libip.compute_paths_for_class(
    #             data, 'segm_false', 'conts_false', 'dt_false', 'gt_false',
    #             experiment_params, for_class=False, ignore=[], debug=params['debug'],
    #             logger=yparams
    #         )
    #         paths.dss()
    #
    #         yparams.logging('\nPaths datastructure: \n\n{}', paths.datastructure2string(maxdepth=3))

    # data = ipl()
    # for sourcekey, source in thisparams['sources'].iteritems():
    #
    #     # Load the necessary images
    #     #   1. Determine the settings for fetching the data
    #     try:
    #         recursive_search = False
    #         recursive_search = thisparams['skwargs', 'default', 'recursive_search']
    #         recursive_search = thisparams['skwargs', sourcekey, 'recursive_search']
    #     except KeyError:
    #         pass
    #     if len(source) > 2:
    #         skeys = source[2]
    #     else:
    #         skeys = None
    #
    #     #   2. Load the data
    #     yparams.logging('skeys = {}', skeys)
    #     yparams.logging('recursive_search = {}', recursive_search)
    #     data[sourcekey] = load_images(
    #         params[source[0]] + params[source[1]], skeys=skeys, recursive_search=recursive_search,
    #         logger=yparams
    #     )
    #
    # data['contacts'].reduce_from_leafs(iterate=True)
    # data['disttransf'].reduce_from_leafs(iterate=True)
    #
    # # Set targetfile
    # targetfile = params[thisparams['target'][0]] \
    #              + params[thisparams['target'][1]]
    #
    # yparams.logging('\nInitial datastructure: \n\n{}', data.datastructure2string(maxdepth=3))
    #
    # for d, k, v, kl in data['segmentation'].data_iterator(yield_short_kl=True, leaves_only=True):
    #     yparams.logging('===============================\nWorking on image: {}', kl + [k])
    #
    #     # # TODO: Implement copy full logger
    #     # data[kl].set_logger(data.get_logger())
    #
    #     # prepare the dict for the path computation
    #     indata = ipl()
    #     indata['segmentation'] = np.array(data['segmentation'][kl][k])
    #     indata['contacts'] = np.array(data['contacts'][kl][k])
    #     indata['groundtruth'] = np.array(data['groundtruth'][kl][params['gtruthname']])
    #     indata['disttransf'] = np.array(data['disttransf'][kl][k])
    #     yparams.logging('Input datastructure: \n\n{}', indata.datastructure2string())
    #     # Compute the paths sorted into their respective class
    #     paths = ipl()
    #     paths[kl + [k]] = libip.compute_paths_with_class(
    #         indata, 'segmentation', 'contacts', 'disttransf', 'groundtruth',
    #         thisparams,
    #         ignore=thisparams['ignorelabels'],
    #         max_end_count=thisparams['max_end_count'],
    #         max_end_count_seed=thisparams['max_end_count_seed'],
    #         debug=params['debug']
    #     )
    #
    #     # Write the result to file
    #     paths.write(filepath=targetfile)


def run_compute_paths(yamlfile, logging=True):

    yparams = YamlParams(filename=yamlfile)
    params = yparams.get_params()

    # Logger stuff
    yparams.set_indent(1)
    yparams.startlogger(
        filename=params['resultfolder'] + 'compute_paths.log',
        type='w', name='ComputePaths'
    )

    try:

        compute_paths(yparams)

        yparams.logging('')
        yparams.stoplogger()

    except:
        raise
        yparams.errout('Unexpected error')


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters_ref.yml'

    run_compute_paths(yamlfile, logging=False)