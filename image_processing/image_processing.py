
import h5py
import numpy as np
import re
import vigra
import scipy
from scipy import ndimage
import random

__author__ = 'jhennies'

class ImageProcessing:

    _image = None

    def __init__(self, image):
        self._image = image
        if image is None:
            print "ImageProcessing: Empty construction."

    @classmethod
    def empty(cls):
        return cls(None)

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

    def resize(self, zoom, mode):
        # self._image = vigra.sampling.resizeImageNoInterpolation(self._image, shape=shape)
        print 'Resizing with mode = ' + mode
        self._image = scipy.ndimage.interpolation.zoom(self._image, zoom, mode=mode, order=0)
        # scipy.misc.imresize(self._image, shape, interp='nearest')

    def getlabel(self, label):
        self._image = (self._image == label).astype(np.uint32)

    def anytask(self, function, *args, **kwargs):
        self._image = function(self._image, *args, **kwargs)

    def amax(self):
        return np.amax(self._image)

class ImageFileProcessing:

    _imagePath = None
    _imageFile = None
    _imageFileName = None
    _imageName = None
    _imageID = 0
    _data = None
    _boundaries = None

    def __init__(self, image_path, image_file, image_name, image_id):
        self.set_file(image_path, image_file, image_name, image_id)
        self.load_h5()

    @classmethod
    def empty(cls):
        """ Empty construction is intended for debugging purposes """
        return cls(None, None, None, None)

    def set_file(self, image_path, image_file, image_name, image_id):
        self._imagePath = image_path
        self._imageFile = image_file
        self._imageName = image_name
        self._imageID = image_id
        if self._imageFile is not None:
            self._imageFileName = re.sub('\.h5$', '', self._imageFile)

    def set_image(self, image):
        self._data.set_image(image)

    def load_h5(self, im_file=None, im_id=None, im_name=None):

        if self._imagePath is not None:
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

        else:
            print 'ImageFileProcessing: Empty construction.'
            self._data = ImageProcessing.empty()
            return None

    def get_image(self):
        return self._data.get_image()

    def get_filename(self):
        return self._imageFileName + '.h5'

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

    def resize(self, zoom, mode):
        self._data.resize(zoom, mode)
        self._imageFileName += '.resize'

    def getlabel(self, label):
        self._data.getlabel(label)
        self._imageFileName += '.lbl_' + str(label)

    def anytask(self, function, addtofilename, *args, **kwargs):
        """
        :type function
        :param function:
            A function in the form: func(image, *args, **kwargs)
            Note that the first parameter is fixed to the currently loaded image

        :type str
        :param addtofilename:
            Extension to the output file name; defaults to '.modified' when set to None
            If no extension is desired supply as empty string ('')
        """
        self._data.anytask(function, *args, **kwargs)
        if addtofilename is not None:
            self._imageFileName += addtofilename
        else:
            self._imageFileName += '.modified'

    def amax(self):
        return self._data.amax()

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

    ### EXAMPLE ###

    # Create object and define image file
    ifp = ImageFileProcessing(
        "/media/julian/Daten/neuraldata/isbi_2013/data/",
        "probabilities_test.h5", None, 0)

    # Modify the image...

    # ... with some pre-defined operations
    ifp.invert_image()
    ifp.swapaxes(0, 2)
    ifp.rollaxis(1, 0)

    # ... with an arbitrary function
    def plusx(array, x):
        return array + x
    ifp.anytask(plusx, '.plus5', 5)

    # Getter functions
    print ifp.get_image().shape
    print ifp.get_filename()

    # # Write the result (File name is automatically generated depending on the performed operations)
    # ifp.write()


