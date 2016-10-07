import h5py
import numpy as np
import re
import vigra
import scipy
from scipy import ndimage, misc
from skimage.morphology import watershed
import time
# import random
import copy
import yaml

__author__ = 'jhennies'


class Hdf5Processing:

    _data = None

    def __init__(self, data=None, dataname=None):
        if data is not None:
            self.setdata(data, dataname)

    def setdata(self, data, append=True):

        if not append or self._data is None:
            self._data = {}

        for d in data:
            self._data[d] = data[d]

    def write(self, filepath):

        of = h5py.File(filepath)

        for dk, dv in self._data.items():

            if type(dv) is list:

                grp = of.create_group(str(dk))
                for i in xrange(0, len(dv)):
                    grp.create_dataset(str(i), data=dv[i])

            elif type(dv) is np.array:

                of.create_dataset(dk, data=dv)

            else:
                print 'Warning in Hdf5Processing.write(): Nothing to write.'

        of.close()

if __name__ == '__main__':
    pass