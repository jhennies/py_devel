
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


def simplify_statistics(statistics, iterations=3):

    newstats = statistics.dcp()

    for i in xrange(0, iterations):
        for d, k, v, kl in statistics.data_iterator(yield_short_kl=True):
            if v == 0 or not v:
                newstats[kl].pop(k)

        statistics = newstats.dcp()

    return newstats


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

        statistics = rdict()

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
            paths[exp_lbl][exp_class_lbl], statistics[exp_lbl][exp_class_lbl] = libip.compute_paths_for_class(
                data, 'segm', 'conts', 'dt', 'gt',
                exp_params, for_class=for_class, ignore=[], debug=all_params['debug'],
                logger=yparams
            )

            yparams.logging(
                '\nPaths datastructure after running {}: \n\n{}',
                exp_class_lbl,
                paths.datastructure2string()
            )

            def val(x):
                return x

            yparams.logging(
                '\nStatistics after {}: \n\n{}', exp_class_lbl,
                simplify_statistics(statistics[exp_lbl]).datastructure2string(function=val)
            )

            # Save the result to disk
            # -----------------------
            targetfile = all_params[exp_target[0]] + all_params[exp_target[1]]
            paths.write(filepath=targetfile)

        def val(x):
            return x
        yparams.logging(
            '\nStatistics after full experiment: \n\n{}',
            simplify_statistics(statistics[exp_lbl]).datastructure2string(function=val)
        )


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

        yparams.errout('Unexpected error')


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters_ref.yml'

    run_compute_paths(yamlfile, logging=False)