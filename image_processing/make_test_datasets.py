
import h5py
import numpy as np
import re
from scipy import ndimage
import random
# import scipy
from netdatautils import fromh5, toh5
import pickle
import cPickle as pkl

from make_membrane_label import ImageFileProcessing

__author__ = 'jhennies'

class modify_datasets():

    def __init__(self):
        pass

    def copy_h5_dataset(self, sourcepath, h5name, targetpath, dataslice=None, targeth5name=None):

        data = fromh5(sourcepath, h5name, dataslice=dataslice, asnumpy=True)

        if targeth5name is None:
            targeth5name = h5name

        toh5(data, targetpath, targeth5name)


    def make_h5_zeros(self, path, h5name, size):

        ifp = ImageFileProcessing()
        data = np.zeros(size, dtype=int)
        ifp.write_h5(path, data, h5name)

    def open_rois(self, input="/media/julian/Daten/mobi/h1.hci/data/fib25/rois.pkl"):
        f = open(input)
        data = pickle.load(f)
        return data


    def rois2slices(self, path, targetpath, cubesize=512):

        rois = self.open_rois(input=path)
        rois = zip(*[iter(rois)]*3)
        print len(rois)

        slices = tuple()
        # Iterate over ROIs
        for roi in rois:
            # print roi
            # Get the slice information
            sl = (slice(roi[0], roi[0]+cubesize),
                  slice(roi[1], roi[1]+cubesize),
                  slice(roi[2], roi[2]+cubesize))

            print "slice:"
            print sl

            slices += (sl,)

            # # Load the respective part of the image
            # data = self.load_data(sl)
            # # print data[0:10, 0:10, 0:10]

            # # Compute the neuronal network
            # self.nn_on_cube(input=data, resultfile=self._resultfile, cubename=str(roi[0]) + "_" + str(roi[1]) + "_" + str(roi[2]))


        print slices

        pickle.dump(slices, open(targetpath, "wb"), protocol=pkl.HIGHEST_PROTOCOL)

    def filter_rois(self, path, targetpath, dataslice, adjust_indices=True):

        rois = self.open_rois(path)
        print len(rois)
        # rois = filter((lambda x: [el if el[0] > 512 for el in rois]), rois)
        for i in xrange(0,3):
            rois = filter(lambda x: x[i].start >= dataslice[i].start, rois)
            rois = filter(lambda x: x[i].stop < dataslice[i].stop, rois)
        # rois = [x for x in rois if x[0] > 1536]
        print len(rois)
        # print rois

        self.save_as_pickle(rois, targetpath)

        pass

    def save_as_pickle(self, object, targetpath):
        pickle.dump(object, open(targetpath, "wb"), protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    # dataslice = (slice(512, 1536), slice(512, 1536), slice(512, 1536))
    dataslice = (slice(2048, 3072), slice(416, 1440), slice(1408, 3328))
    md = modify_datasets()
    # md.copy_h5_dataset('/media/julian/Daten/mobi/h1.hci/data/fib25/raw_fib25.h5', 'data',
    #                    '/media/julian/Daten/mobi/h1.hci/data/fib25/raw_fib25_crop1024.h5',
    #                    dataslice=dataslice)

    # md.rois2slices('/media/julian/Daten/mobi/h1.hci/data/fib25/rois512.pkl',
    #                '/media/julian/Daten/mobi/h1.hci/data/fib25/roissl512.pkl', cubesize=512)

    md.filter_rois('/media/julian/Daten/mobi/h1.hci/data/fib25/roissl512.pkl',
                   '/media/julian/Daten/mobi/h1.hci/data/fib25/roissl512_crop1024.pkl',
                   dataslice=dataslice, adjust_indices=True)