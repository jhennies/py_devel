
import h5py
import numpy as np
import re
from scipy import ndimage
import random
# import scipy

from make_membrane_label import ImageFileProcessing

__author__ = 'jhennies'


if __name__ == "__main__":

    ifp = ImageFileProcessing()
    # ifp.set_file('/media/julian/Daten/mobi/h1.hci/data/testdata', "zeros.h5", "zeros", 0)

    data = np.zeros((256, 256, 256), dtype=int)

    ifp.write_h5('/media/julian/Daten/mobi/h1.hci/data/testdata/zeros.h5', data, 'zeros')
