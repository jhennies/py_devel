
import h5py
import numpy as np
import re
from scipy import ndimage
import random

__author__ = 'jhennies'

class ImageProcessing:

    _image = None

    def __init__(self):
        pass

    def __init__(self, image):
        self._image = image

    def set_image(self, image):
        self._image = image

    def get_image(self):
        return self._image

    def invert_image(self):
        self._image = np.amax(self._image) - self._image

    def swapaxes(self, axis1, axis2):
        self._image = np.swapaxes(self._image, axis1, axis2)

    def rollaxis(self, axis, start=0):
        self._image = np.rollaxis(self._image, axis, start)


class ImageFileProcessing:

    _imagePath = ''
    _imageFile = ''
    _imageFileName = ''
    _imageName = None
    _imageID = 0
    _data = None
    _boundaries = None

    def __init__(self, image_path, image_file, image_name, image_id):
        self.set_file(image_path, image_file, image_name, image_id)
        self.load_h5()

    def set_file(self, image_path, image_file, image_name, image_id):
        self._imagePath = image_path
        self._imageFile = image_file
        self._imageName = image_name
        self._imageID = image_id
        self._imageFileName = re.sub('\.h5$', '', self._imageFile)

    def load_h5(self, im_file=None, im_id=None, im_name=None):

        if im_file is None:
            f = h5py.File(self._imagePath + self._imageFile)
            if self._imageName is None:
                self._imageName = f.keys()[self._imageID]
            self._data = ImageProcessing(np.array(f.get(self._imageName)))
            f.close()
            return self._data
        else:
            print "Loading file..."
            print im_file
            f = h5py.File(im_file)
            if im_name is None:
                im_name = f.keys()[im_id]
            data = np.array(f.get(im_name))
            f.close()
            return data

    ###########################################################################################
    # Image processing

    def invert_image(self):
        self._data.invert_image()
        self._imageFileName += '.inv'

    def swapaxes(self, axis1, axis2):
        self._data.swapaxes(axis1, axis2)
        self._imageFileName += '.swpxs_' + str(axis1) + '_' + str(axis2)

    def rollaxis(self, axis, start=0):
        self._data.rollaxis(axis, start)
        self._imageFileName += '.rllxs_' + str(axis) + '_' + str(start)

    ###########################################################################################
    # Write h5 files

    def write_h5(self, nfile, data, image_name=None):
        print "Writing..."
        of = h5py.File(self._imagePath + nfile)
        if image_name is None:
            of.create_dataset(self._imageName, data=data)
        else:
            of.create_dataset(image_name, data=data)
        of.close()

    def write(self):
        self.write_h5(self._imageFileName + '.h5', self._data.get_image())


if __name__ == "__main__":

    # Create object and define image file
    ifp = ImageFileProcessing(
        "/media/julian/Daten/neuraldata/isbi_2013/data/",
        "probabilities_test.h5", None, 0)

    # Modify the image
    ifp.invert_image()
    ifp.swapaxes(0, 2)
    ifp.rollaxis(1, 0)

    # Write the result (File name is automatically generated depending on the performed operations)
    ifp.write()
