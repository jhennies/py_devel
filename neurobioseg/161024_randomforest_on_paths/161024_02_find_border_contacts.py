
from hdf5_image_processing import Hdf5ImageProcessingLib as IPL
import random
import vigra.graphs as graphs
import numpy as np
import os
import inspect
from shutil import copy
import processing_lib as lib
from copy import deepcopy

__author__ = 'jhennies'


def find_orphans(hfp):
    pass


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'

    hfp = IPL(
        yaml=yamlfile,
        yamlspec={'path': 'datafolder', 'filename': 'largeobjfile', 'skeys': 'largeobjname'},
        tkeys='largeobj',
        castkey=None
    )
    params = hfp.get_params()
    hfp.startlogger(filename=params['resultfolder'] + 'find_orphans.log', type='w')

    try:

        # Copy the script file and the parameters to the scriptsfolder
        copy(inspect.stack()[0][1], params['scriptsfolder'])
        copy(yamlfile, params['scriptsfolder'] + 'find_orphans.parameters.yml')
        # Write script and parameters to the logfile
        hfp.code2log(inspect.stack()[0][1])
        hfp.logging('')
        hfp.yaml2log()
        hfp.logging('')

        hfp.logging('\nhfp datastructure: \n---\n{}', hfp.datastructure2string(maxdepth=1))

        find_orphans(hfp)

        # TODO: Comment in when ready
        # hfp.write(filepath=params['intermedfolder'] + params['orphansfile'])

        hfp.logging('\nFinal dictionary structure:\n---\n{}', hfp.datastructure2string())
        hfp.logging('')
        hfp.stoplogger()

    except:

        hfp.errout('Unexpected error')