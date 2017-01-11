
from hdf5_image_processing import Hdf5ImageProcessingLib as ipl
import numpy as np
import os
from hdf5_processing import RecursiveDict as rdict
import processing_libip as libip
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


def find_border_contacts(yparams):

    params = yparams.get_params()
    thisparams = rdict(params['find_border_contacts'])
    # targetfile = params['intermedfolder'] + params['featureimsfile']

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
        data[sourcekey] = load_images(
            params[source[0]] + params[source[1]], skeys=skeys, recursive_search=recursive_search,
            logger=yparams
        )

    data.reduce_from_leafs(iterate=True)

    # Set targetfile
    targetfile = params[thisparams['target'][0]] \
                 + params[thisparams['target'][1]]

    yparams.logging('\nInitial datastructure: \n\n{}', data.datastructure2string(maxdepth=3))

    for d, k, v, kl in data['segmentation'].data_iterator(yield_short_kl=True, leaves_only=True):

        yparams.logging('===============================\nWorking on image: {}', kl + [k])

        # # TODO: Implement copy full logger
        # data[kl].set_logger(data.get_logger())

        # We need: the distance transform of the MERGED labels (i.e. segmentation) and the
        #   corresponding segmentation
        data['segmentation'][kl][k] = libip.find_border_contacts_arr(
            data['segmentation'][kl][k],
            data['disttransf'][kl][k],
            tkey=params['borderctname'],
            debug=params['debug']
        )

        # Write the result to file
        data['segmentation'].write(filepath=targetfile, keys=[kl + [k]])
        # Free memory
        # data[kl][k] = None


def run_find_border_contacts(yamlfile, logging=True):

    yparams = YamlParams(filename=yamlfile)
    params = yparams.get_params()

    # Logger stuff
    yparams.set_indent(1)
    yparams.startlogger(
        filename=params['resultfolder'] + 'find_border_contacts.log',
        type='w', name='FindBorderContacts'
    )

    try:

        find_border_contacts(yparams)

        yparams.logging('')
        yparams.stoplogger()

    except:

        yparams.errout('Unexpected error')


if __name__ == '__main__':
    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters_ref.yml'

    run_find_border_contacts(yamlfile, logging=False)