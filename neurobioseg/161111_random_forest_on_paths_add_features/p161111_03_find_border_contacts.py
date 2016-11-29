
from hdf5_image_processing import Hdf5ImageProcessingLib as IPL
import random
import vigra.graphs as graphs
import numpy as np
import os
import inspect
from shutil import copy
import processing_lib as lib
from copy import deepcopy
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
import vigra
from skimage import morphology
from hdf5_processing import RecursiveDict as rdict
import processing_libip as libip

__author__ = 'jhennies'


def find_border_contacts_image_iteration(ipl):
    params = ipl.get_params()
    thisparams = params['find_border_contacts']

    if thisparams['return_bordercontact_images']:
        bordercontacts = IPL()

    for d, k, v, kl in ipl.data_iterator(yield_short_kl=True):

        if k == params['largeobjname']:

            if thisparams['return_bordercontact_images']:
                ipl[kl].setlogger(ipl.getlogger())
                [ipl[kl], bordercontacts[kl]] = find_border_contacts(ipl[kl], (params['largeobjname'], params['largeobjmnames'][0]),
                                     thisparams)
            else:
                ipl[kl].setlogger(ipl.getlogger())
                ipl[kl] = find_border_contacts(ipl[kl], (params['largeobjname'], params['largeobjmnames'][0]),
                                     thisparams)

    if thisparams['return_bordercontact_images']:
        return bordercontacts
    else:
        return None

def load_images(ipl):
    """
    We need to load these images:
    largeobj (the labels)
    largeobjm (merged labels)
    disttransf (distance transform of labels)
    disttransfm (distance transform of merged labels)
    """

    params = ipl.get_params()

    # Load distance transforms
    ipl.data_from_file(
        filepath=params['intermedfolder'] + params['featureimsfile'],
        skeys='disttransf',
        recursive_search=True, nodata=True
    )
    ipl.setdata(ipl.subset('raw', search=True))
    ipl.setdata(ipl.switch_levels(2, 3))
    ipl.reduce_from_leafs()

    # Load largeobj
    ipl.data_from_file(
        filepath=params['intermedfolder'] + params['largeobjfile'],
        skeys=params['largeobjname'],
        recursive_search=True, nodata=True
    )

    # Load largeobjm
    ipl.data_from_file(
        filepath=params['intermedfolder'] + params['largeobjmfile'],
        skeys=params['largeobjmnames'][0],
        recursive_search=True, nodata=True
    )


def find_border_contacts(ipl):

    params = ipl.get_params()
    thisparams = params['find_border_contacts']
    targetfile = params['intermedfolder'] + params['borderctfile']

    # Load the necessary images
    load_images(ipl)

    ipl.logging('\nInitial datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

    maxd = ipl.maxdepth()
    for d, k, v, kl in ipl.data_iterator(maxdepth=ipl.maxdepth() - 2):

        if d == maxd - 2:

            ipl.logging('===============================\nWorking on group: {}', kl)

            # TODO: Implement copy full logger
            ipl[kl].set_logger(ipl.get_logger())

            # # Load the image data into memory
            # ipl[kl].populate(k)

            libip.find_border_contacts(
                ipl[kl],
                (params['largeobjname'], params['largeobjmnames'][0]),
                params['borderctname'],
                debug=False
            )

            # Write the result to file
            ipl.write(filepath=targetfile, keys=[kl + [params['borderctname']]])
            # # Free memory (With this command the original reference to the source file is restored)
            # ipl[kl].unpopulate()


def run_find_border_contacts(yamlfile):

    ipl = IPL(yaml=yamlfile)

    ipl.set_indent(1)

    params = rdict(data=ipl.get_params())
    ipl.startlogger(filename=params['resultfolder'] + 'find_border_contacts.log', type='w', name='FindBorderContacts')

    try:

        # # Copy the script file and the parameters to the scriptsfolder
        # copy(inspect.stack()[0][1], params['scriptsfolder'])
        # copy(yamlfile, params['scriptsfolder'] + 'find_border_contacts.parameters.yml')

        # ipl.logging('\nInitial datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

        find_border_contacts(ipl)

        # ipl.logging('\nFinal datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

        # ipl.write(filepath=params['intermedfolder'] + params['largeobjfile'])

        ipl.logging('')
        ipl.stoplogger()

    except:

        ipl.errout('Unexpected error')


if __name__ == '__main__':
    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'

    run_find_border_contacts(yamlfile)