
import os
import inspect
# from hdf5_image_processing import Hdf5ImageProcessing as IP, Hdf5ImageProcessingLib as IPL
from hdf5_slim_processing import Hdf5Processing as hp
from shutil import copy, copyfile
import numpy as np
# import matplotlib.pyplot as plt
# import processing_libip as libip
import slim_processing_libhp as libhp
import sys
# from simple_logger import SimpleLogger
from yaml_parameters import YamlParams
from concurrent import futures


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


def remove_small_objects(yparams):
    """
    :param ipl: A Hdf5ImageProcessingLib instance containing labelimages named as specified in ipl.get_params()['labelsname']

    ipl.get_params()

        remove_small_objects
            bysize
            relabel

        largeobjname

        labelsname

    :param key: the source key for calculation
    """

    params = yparams.get_params()
    thisparams = params['remove_small_objects']

    # Make dictionary of all sources
    all_data = hp()
    for i in xrange(0, len(thisparams['sources'])):

        kwargs = None
        if len(thisparams['sources'][i]) > 2:
            kwargs = thisparams['sources'][i][2]
        all_data[i] = load_images(
            params[thisparams['sources'][i][0]] + params[thisparams['sources'][i][1]],
            logger=yparams, **kwargs
        )

    # Process all data items
    def filtering_wrapper(d, k, v, kl):
        yparams.logging('===============================\nWorking on image: {}', kl)

        targetfile = params[thisparams['targets'][kl[0]][0]] + params[thisparams['targets'][kl[0]][1]]

        parallelize_filtering = False
        if thisparams['filtering_threads'] > 1:
            parallelize_filtering = True

        result = hp()
        result[kl[1:]] = libhp.remove_small_objects_relabel(
            np.array(v), thisparams['bysize'],
            relabel=thisparams['relabel'],
            consecutive_labels=thisparams['consecutive_labels'],
            parallelize=parallelize_filtering, max_threads=thisparams['filtering_threads'],
            logger=yparams
        )

        # Write the result to file
        result.write(filepath=targetfile)

        return result[kl[1:]]

    if thisparams['image_threads'] > 1:

        with futures.ThreadPoolExecutor(thisparams['image_threads']) as filter_small:

            tasks = hp()
            for d, k, v, kl in all_data.data_iterator(leaves_only=True):
                tasks[kl] = filter_small.submit(filtering_wrapper, d, k, v, kl)

        # for d, k, v, kl in tasks.data_iterator(leaves_only=True):
        #     result = v.result()

    else:
        for d, k, v, kl in all_data.data_iterator(leaves_only=True):
            filtering_wrapper(d, k, v, kl)

    # Close the source files
    for k, v in all_data.iteritems():
        v.close()


def run_remove_small_objects(yamlfile):

    yparams = YamlParams(filename=yamlfile)
    params = yparams.get_params()

    # Logger stuff
    yparams.set_indent(1)
    yparams.startlogger(
        filename=params['resultfolder'] + 'remove_small_objects.log',
        type='w', name='RemoveSmallObjects'
    )

    try:

        remove_small_objects(yparams)

        yparams.logging('')
        yparams.stoplogger()

    except:

        yparams.errout('Unexpected error')


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters_ref.yml'

    run_remove_small_objects(yamlfile)

