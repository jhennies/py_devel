
import os
import inspect
from hdf5_image_processing import Hdf5ImageProcessing as IP, Hdf5ImageProcessingLib as IPL
from shutil import copy, copyfile
import numpy as np
import matplotlib.pyplot as plt


__author__ = 'jhennies'


def accumulate_small_objects(ipl, key, thisparams):

    size_exclusion = thisparams['bysize']
    ipl.logging('size_exclusion = {}', size_exclusion)

    unique, counts = np.unique(ipl[key], return_counts=True)
    ipl.logging('unique = {}, counts = {}', unique, counts)

    ipl.logging('{}', unique[counts < size_exclusion])

    # With accumulate set to True, this iterator does everything we need:
    # Each label with a count larger than size_exclusion is added to lblim which is initialized as np.zeros(...)
    for lbl, lblim in ipl.label_image_iterator(key=key,
                                               labellist=unique[counts > size_exclusion],
                                               accumulate=True, relabel=thisparams['relabel']):
        ipl.logging('---\nIncluding label {}', lbl)

    del ipl[key]

    ipl[params['largeobjname']] = lblim

    return ipl


def remove_small_objects(ipl):
    """
    :param hfp: A Hdf5ImageProcessing instance containing a labelimage named 'labels'

    hfp.get_params()

        remove_small_objects
            bysize
            relabel

        largeobjname

    :param key: the source key for calculation
    """

    params = ipl.get_params()
    thisparams = params['remove_small_objects']

    for d, k, v, kl in ipl.data_iterator(yield_short_kl=True):

        if k == params['labelsname']:

            ipl.logging('===============================\nWorking on image: {}', kl + [k])

            ipl[kl].setlogger(ipl.getlogger())
            ipl[kl] = accumulate_small_objects(ipl[kl], k, thisparams)


if __name__ == '__main__':

    resultsfolder = '/mnt/localdata02/jhennies/neuraldata/results/cremi_2016/161110_random_forest_of_paths/'

    yamlfile = resultsfolder + '/parameters.yml'

    ipl = IPL(
        yaml=yamlfile,
        yamlspec={'path': 'datafolder', 'filename': 'labelsfile'}
    )
    params = ipl.get_params()
    ipl.startlogger(filename=params['resultfolder']+'remove_small_objects.log', type='a')

    try:

        # Create folder for scripts
        if not os.path.exists(params['scriptsfolder']):
            os.makedirs(params['scriptsfolder'])
        else:
            if params['overwriteresults']:
                ipl.logging('remove_small_objects: Warning: Scriptsfolder already exists and content will be overwritten...\n')
            else:
                raise IOError('remove_small_objects: Error: Scriptsfolder already exists!')

        # Create folder for intermediate results
        if not os.path.exists(params['intermedfolder']):
            os.makedirs(params['intermedfolder'])
        else:
            if params['overwriteresults']:
                ipl.logging('remove_small_objects: Warning: Intermedfolder already exists and content will be overwritten...\n')
            else:
                raise IOError('remove_small_objects: Error: Intermedfolder already exists!')

        # Copy the script file and the parameters to the scriptsfolder
        copy(inspect.stack()[0][1], params['scriptsfolder'])
        copy(yamlfile, params['scriptsfolder'] + 'remove_small_objects.parameters.yml')
        # Write script and parameters to the logfile
        ipl.code2log(inspect.stack()[0][1])
        ipl.logging('')
        ipl.yaml2log()
        ipl.logging('')

        ipl.logging('\nipl datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

        remove_small_objects(ipl)

        ipl.logging('\nFinal datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

        ipl.write(filepath=params['intermedfolder'] + params['largeobjfile'])

        ipl.logging('')
        ipl.stoplogger()

    except:

        ipl.errout('Unexpected error')

