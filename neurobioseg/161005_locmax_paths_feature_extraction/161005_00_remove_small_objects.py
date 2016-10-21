
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
    hfp.startlogger(filename=params['resultfolder']+'remove_small_objects.log', type='a')

    try:

        # Create folder for scripts
        if not os.path.exists(params['scriptsfolder']):
            os.makedirs(params['scriptsfolder'])
        else:
            if params['overwriteresults']:
                hfp.logging('remove_small_objects: Warning: Scriptsfolder already exists and content will be overwritten...\n')
            else:
                raise IOError('remove_small_objects: Error: Scriptsfolder already exists!')

        # Create folder for intermediate results
        if not os.path.exists(params['intermedfolder']):
            os.makedirs(params['intermedfolder'])
        else:
            if params['overwriteresults']:
                hfp.logging('remove_small_objects: Warning: Intermedfolder already exists and content will be overwritten...\n')
            else:
                raise IOError('remove_small_objects: Error: Intermedfolder already exists!')

        # Copy the script file and the parameters to the scriptsfolder
        copy(inspect.stack()[0][1], params['scriptsfolder'])
        copy(yamlfile, params['scriptsfolder'] + 'remove_small_objects.parameters.yml')
        # Write script and parameters to the logfile
        hfp.code2log(inspect.stack()[0][1])
        hfp.logging('')
        hfp.yaml2log()
        hfp.logging('')

        hfp.logging('\nhfp datastructure: \n\n{}', hfp.datastructure2string(maxdepth=1))

        remove_small_objects(hfp)

        hfp.write(filepath=params['intermedfolder'] + params['largeobjfile'])

        hfp.logging('')
        hfp.stoplogger()

    except:

        hfp.errout('Unexpected error')

