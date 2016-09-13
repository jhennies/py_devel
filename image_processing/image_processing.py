
import h5py
import numpy as np
import re
import vigra
import scipy
from scipy import ndimage, misc
import random
import copy

__author__ = 'jhennies'


# _____________________________________________________________________________________________
# The image processing library

def invert_image(image):
    return np.amax(image) - image


def swapaxes(image, axis1, axis2):
    return np.swapaxes(image, axis1, axis2)


def rollaxis(image, axis, start=0):
    return np.rollaxis(image, axis, start)


def resize(image, zoom, mode):
    # self._image = vigra.sampling.resizeImageNoInterpolation(self._image, shape=shape)
    print 'Resizing with mode = ' + mode
    return scipy.ndimage.interpolation.zoom(image, zoom, mode=mode, order=0)
    # scipy.misc.imresize(self._image, shape, interp='nearest')


def resize_z_nearest(image, z):
    img = image
    newimg = np.zeros((img.shape[0], img.shape[1], z))
    for i in xrange(0, img.shape[1]):
        t = img[:, i, :]
        # print img.shape
        # print t.shape
        # t = np.swapaxes(t, 1, 2)
        misc.imresize(t, z, interp='nearest')
        # t = np.swapaxes(t, 1, 2)

        newimg[:, i, :] = t

    return newimg


def getlabel(image, label):
    return (image == label).astype(np.uint32)


def amax(image):
    return np.amax(image)

# _____________________________________________________________________________________________


class ImageProcessing:

    _image = None

    def __init__(self, image):
        self._image = image
        if image is None:
            print "ImageProcessing: Empty construction."

    @classmethod
    def empty(cls):
        return cls(None)

    ###########################################################################################
    # Data operations

    def set_image(self, image):
        self._image = image

    def get_image(self):
        return self._image

    def converttodict(self, name):
        if type(self._image) is dict:
            print 'Warning: Type already a dict!'
            return
        else:
            t = self._image
            self._image = {}
            self._image[name] = t

    def addtodict(self, image, name):
        if type(self._image) is dict:
            self._image[name] = image
        else:
            print 'Warning: Not like this, convert to dict first!'

    def deepcopy(self, sourcekey, targetkey):
        if type(self._image) is dict:
            self._image[targetkey] = copy.deepcopy(self._image[sourcekey])
        else:
            print 'Warning: Deepcopy only implemented for dict type!'

    ###########################################################################################
    # Image processing

    def anytask(self, function, ids, *args, **kwargs):
        if type(self._image) is dict:
            if ids is None:
                for id in self._image:
                    self._image[id] = function(self._image[id], *args, **kwargs)
            else:
                for id in ids:
                    self._image[id] = function(self._image[id], *args, **kwargs)
        else:
            self._image = function(self._image, *args, **kwargs)

    def anytask_rtrn(self, function, ids, *args, **kwargs):
        if type(self._image) is dict:
            returndict = {}
            if ids is None:
                for id in self._image:
                    returndict[id] = function(self._image[id], *args, **kwargs)
            else:
                for id in ids:
                    returndict[id] = function(self._image[id], *args, **kwargs)
            return returndict
        else:
            return function(self._image, *args, **kwargs)

    def invert_image(self, ids=None):
        self.anytask(invert_image, ids)

    def swapaxes(self, axis1, axis2, ids=None):
        self.anytask(swapaxes, ids, axis1, axis2)

    def rollaxis(self, axis, start=0, ids=None):
        self.anytask(rollaxis, ids, axis, start=start)

    def resize(self, zoom, mode, ids=None):
        self.anytask(resize, ids, zoom, mode)

    def resize_z_nearest(self, z, ids=None):
        self.anytask(resize_z_nearest, ids, z)

    def getlabel(self, label, ids=None):
        self.anytask(getlabel, ids, label)

    def amax(self, ids=None):
        return self.anytask_rtrn(amax, ids=ids)


# _____________________________________________________________________________________________


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

    def get_filename(self):
        return self._imageFileName + '.h5'

    def addtoname(self, addstr):
        self._imageFileName += addstr

    def deepcopy(self, sourcekey, targetkey):
        self._data.deepcopy(sourcekey, targetkey)

    ###########################################################################################
    # Data operations

    def set_image(self, image):
        self._data.set_image(image)

    def get_image(self):
        return self._data.get_image()

    def converttodict(self, name):
        self._data.converttodict(name)

    def addtodict(self, image, name):
        self._data.addtodict(image, name)

    ###########################################################################################
    # Image processing

    def anytask(self, function, addtofilename, ids, *args, **kwargs):
        """
        :type function
        :param function:
            A function in the form: func(image, *args, **kwargs)
            Note that the first parameter is fixed to the currently loaded image

        :type str
        :param addtofilename:
            Extension to the output file name; defaults to '.modified' when set to None
            If no extension is desired supply as empty string ('')

        :type list
        :param ids:
            list of keys denoting the dictionary entries in self._data._image which will be processed
            Set to None if self._data is not a dictionary or for processing of all entries
        """
        self._data.anytask(function, ids, *args, **kwargs)
        if addtofilename is not None:
            self._imageFileName += addtofilename
        else:
            self._imageFileName += '.modified'

    def anytask_rtrn(self, function, ids, *args, **kwargs):
        """
        :type function
        :param function:
            A function in the form: func(image, *args, **kwargs)
            Note that the first parameter is fixed to the currently loaded image

        :type list
        :param ids:
            list of keys denoting the dictionary entries in self._data._image which will be processed
            Set to None if self._data is not a dictionary or for processing of all entries
        """
        return self._data.anytask_rtrn(function, ids, *args, **kwargs)

    def invert_image(self, ids=None):
        self._data.invert_image(ids=ids)
        self._imageFileName += '.inv'

    def swapaxes(self, axis1, axis2, ids=None):
        self._data.swapaxes(axis1, axis2, ids=ids)
        self._imageFileName += '.swpxs_' + str(axis1) + '_' + str(axis2)

    def rollaxis(self, axis, start=0, ids=None):
        self._data.rollaxis(axis, start, ids=ids)
        self._imageFileName += '.rllxs_' + str(axis) + '_' + str(start)

    def resize(self, zoom, mode, ids=None):
        self._data.resize(zoom, mode, ids=ids)
        self._imageFileName += '.resize'

    def resize_z_nearest(self, z, ids=None):
        self._data.resize_z_nearest(z, ids=ids)
        self._imageFileName += '.resizez_' + str(z)

    def getlabel(self, label, ids=None):
        self._data.getlabel(label, ids=ids)
        self._imageFileName += '.lbl_' + str(label)

    def amax(self, ids=None):
        return self._data.amax(ids=ids)

    ###########################################################################################
    # Write h5 files

    def write_h5(self, nfile, data, image_names=None, dict_ids=None):
        print "Writing..."
        of = h5py.File(self._imagePath + nfile)

        if type(data) is dict:

            if dict_ids is None:
                for d in data:
                    if image_names is None:
                        of.create_dataset(d, data=data[d])
                    else:
                        of.create_dataset(image_names[d], data=data[d])
            else:
                for d in dict_ids:
                    if image_names is None:
                        of.create_dataset(d, data=data[d])
                    else:
                        of.create_dataset(image_names[d], data=data[d])

        else:

            if image_names is None:
                of.create_dataset(self._imageName, data=data)
            else:
                of.create_dataset(image_names, data=data)

        of.close()

    def write(self, dict_ids=None, filename=None):
        """
        :type list
        :param dict_ids: Set only if data is a dict!
            Use to specify which entries in the data dictionary are written to file
            Default: None (everything is written)
        """
        if filename is None:
            self.write_h5(self._imageFileName + '.h5', self._data.get_image(), dict_ids=dict_ids)
        else:
            self.write_h5(filename, self._data.get_image(), dict_ids=dict_ids)

# _____________________________________________________________________________________________


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


