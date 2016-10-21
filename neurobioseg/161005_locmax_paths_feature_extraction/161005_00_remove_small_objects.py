
import os
import inspect
from hdf5_image_processing import Hdf5ImageProcessing as IP, Hdf5ImageProcessingLib as IPL
from shutil import copy, copyfile
import numpy as np
import matplotlib.pyplot as plt


__author__ = 'jhennies'


def remove_small_objects(hfp):
    """
    :param hfp: A Hdf5ImageProcessing instance containing a labelimage named 'labels'
    """

    params = hfp.get_params()
    thisparams = params['remove_small_objects']

    size_exclusion = thisparams['bysize']
    hfp.logging('size_exclusion = {}', size_exclusion)

    unique, counts = np.unique(hfp['labels'], return_counts=True)
    hfp.logging('unique = {}, counts = {}', unique, counts)

    hfp.logging('{}', unique[counts < size_exclusion])

    # With accumulate set to True, this iterator does everything we need:
    # Each label with a count larger than size_exclusion is added to lblim which is initialized as np.zeros(...)
    for lbl, lblim in hfp.label_image_iterator(key='labels',
                                               labellist=unique[counts > size_exclusion],
                                               accumulate=True):
        hfp.logging('---\nIncluding label {}', lbl)

    del hfp['labels']
    hfp[params['largeobjname']] = lblim


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'

    hfp = IPL(
        yaml=yamlfile,
        yamlspec={'path': 'datafolder', 'filename': 'labelsfile'},
        tkeys='labels',
        castkey=None
    )
    params = hfp.get_params()
    hfp.startlogger(filename=params['intermedfolder']+'remove_small_objects.log', type='a')

    try:

        # Create new subfolder for intermediate results
        if not os.path.exists(params['intermedfolder'] + 'remove_small_objects'):
            os.makedirs(params['intermedfolder'] + 'remove_small_objects')
        else:
            if params['overwriteresults']:
                hfp.logging('remove_small_objects: Warning: Target already exists and is overwritten...\n')
            else:
                raise IOError('remove_small_objects: Error: Target already exists!')

        # Copy the script file and the parameters to the corresponding intermedfolder
        copy(inspect.stack()[0][1], params['intermedfolder'] + 'remove_small_objects')
        copy(yamlfile, params['intermedfolder'] + 'remove_small_objects/remove_small_objects.parameters.yml')
        # Write script and parameters to the logfile
        hfp.code2log(inspect.stack()[0][1])
        hfp.logging('')
        hfp.yaml2log()
        hfp.logging('')

        hfp.logging('\nhfp datastructure: \n\n{}', hfp.datastructure2string(maxdepth=1))

        remove_small_objects(hfp)

        hfp.write(filepath=params['intermedfolder'] + '/remove_small_objects/' + params['largeobjfile'])

    except:

        hfp.errout('Unexpected error')

