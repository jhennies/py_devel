
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

    params = yparams.get_params()
    thisparams = rdict(params['compute_paths'])

    data = ipl()
    for sourcekey, source in thisparams['sources'].iteritems():

        # Load the necessary images
        #   1. Determine the settings for fetching the data
        try:
            recursive_search = False
            recursive_search = thisparams['skwargs', 'default', 'recursive_search']
            recursive_search = thisparams['skwargs', sourcekey, 'recursive_search']
        except KeyError:
            pass
        if len(source) > 2:
            skeys = source[2]
        else:
            skeys = None

        #   2. Load the data
        yparams.logging('skeys = {}', skeys)
        yparams.logging('recursive_search = {}', recursive_search)
        data[sourcekey] = load_images(
            params[source[0]] + params[source[1]], skeys=skeys, recursive_search=recursive_search,
            logger=yparams
        )

    data['contacts'].reduce_from_leafs(iterate=True)
    data['disttransf'].reduce_from_leafs(iterate=True)

    # Set targetfile
    targetfile = params[thisparams['target'][0]] \
                 + params[thisparams['target'][1]]

    yparams.logging('\nInitial datastructure: \n\n{}', data.datastructure2string(maxdepth=3))

    for d, k, v, kl in data['segmentation'].data_iterator(yield_short_kl=True, leaves_only=True):
        yparams.logging('===============================\nWorking on image: {}', kl + [k])

        # # TODO: Implement copy full logger
        # data[kl].set_logger(data.get_logger())

        # prepare the dict for the path computation
        indata = ipl()
        indata['segmentation'] = np.array(data['segmentation'][kl][k])
        indata['contacts'] = np.array(data['contacts'][kl][k])
        indata['groundtruth'] = np.array(data['groundtruth'][kl][params['gtruthname']])
        indata['disttransf'] = np.array(data['disttransf'][kl][k])
        yparams.logging('Input datastructure: \n\n{}', indata.datastructure2string())
        # Compute the paths sorted into their respective class
        paths = ipl()
        paths[kl + [k]] = libip.compute_paths_with_class(
            indata, 'segmentation', 'contacts', 'disttransf', 'groundtruth',
            thisparams,
            ignore=thisparams['ignorelabels'],
            max_end_count=thisparams['max_end_count'],
            max_end_count_seed=thisparams['max_end_count_seed'],
            debug=params['debug']
        )

        # Write the result to file
        paths.write(filepath=targetfile)


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