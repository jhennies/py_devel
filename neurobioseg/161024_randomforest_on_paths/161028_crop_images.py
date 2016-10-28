
from hdf5_image_processing import Hdf5ImageProcessingLib as IPL
import os
import numpy as np


__author__ = 'jhennies'


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'
    ipl = IPL(
        yaml=yamlfile
    )

    ipl.logging('Parameters: {}', ipl.get_params())
    params = ipl.get_params()

    ipl.data_from_file(filepath=params['datafolder'] + 'cremi.splA.raw_neurons.crop.h5',
                       skeys='raw',
                       tkeys='raw')

    ipl.crop_bounding_rect(np.s_[10:110, 200:712, 200:712], keys='raw')

    ipl.write(filepath=params['datafolder'] + 'cremi.splA.raw_neurons.crop.crop_10-200-200_110-712-712.h5')
