
import os
import inspect
from hdf5_image_processing import Hdf5ImageProcessing as IP, Hdf5ImageProcessingLib as IPL
from shutil import copy, copyfile
import numpy as np
import matplotlib.pyplot as plt
import processing_libip as libip
import sys
from simple_logger import SimpleLogger
from yaml_parameters import YamlParams


__author__ = 'jhennies'


def load_images(filepath, skeys=None, recursive_search=False, logger=None):

    if logger is not None:
        logger.logging('Loading data from \n{}', filepath)
    else:
        print 'Loading data from \n{}'.format(filepath)

    data = IPL()

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

    # sourcefolder = params[thisparams['sourcefolder']]

    for i in xrange(0, len(thisparams['sources'])):

        data = load_images(
            params[thisparams['sources'][i][0]] + params[thisparams['sources'][i][1]],
            skeys=thisparams['kwargs']['skeys'][i],
            recursive_search=thisparams['kwargs']['recursive_search'][i],
            logger=yparams
        )

        targetfile = params[thisparams['targets'][i][0]] + params[thisparams['targets'][i][1]]

        for d, k, v, kl in data.data_iterator(leaves_only=True, yield_short_kl=True):

            yparams.logging('===============================\nWorking on image: {}', kl + [k])

            # Load the image data into memory
            data[kl].populate(k)

            data[kl] = libip.filter_small_objects(
                data[kl], k, thisparams['bysize'], relabel=thisparams['relabel'],
                logger=yparams
            )

            # # Rename the entry
            # ipl[kl].rename_entry(params['labelsname'], params['largeobjname'])

            # Write the result to file
            data.write(filepath=targetfile, keys=[kl + [k]])
            # Free memory (With this command the original reference to the source file is restored)
            data[kl].unpopulate()


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

