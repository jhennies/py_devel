
from image_processing import ImageFileProcessing
import random
import vigra.graphs as graphs
import numpy as np
import os

__author__ = 'jhennies'


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'

    ifp = ImageFileProcessing(
        yaml=yamlfile,
        yamlspec={'image_path': 'intermedfolder', 'image_file': 'largeobjfile', 'image_names': 'largeobjname'},
        asdict=True,
        keys=('largeobj',)
    )
    params = ifp.get_params()
    ifp.addfromfile(params['intermedfolder']+params['largeobjmfile'], image_names=params['largeobjmnames'][0], ids='largeobjm')

    ifp.startlogger(filename=ifp.get_params()['intermedfolder'] + 'locmax_on_disttransf.log', type='a')

    ifp.code2log(__file__)
    ifp.logging('')

    ifp.logging('yamlfile = {}', yamlfile)
    ifp.logging('ifp.get_data().keys() = {}', ifp.get_data().keys())
    ifp.logging('ifp.shape() = {}', ifp.shape())

    ifp.logging('')
    ifp.stoplogger()