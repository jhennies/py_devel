
import h5py
import numpy as np
import re
import vigra
import scipy
from scipy import ndimage, misc
from skimage.morphology import watershed
import time
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


def astype(image, dtype):
    return image.astype(dtype=dtype)


def distance_transform(image, pixel_pitch=()):
    return vigra.filters.distanceTransform(image, pixel_pitch=pixel_pitch)


def filter_values(image, value, type='se', setto=0):

    if type == 's':
        image[image < value] = setto
    elif type == 'se':
        image[image <= value] = setto
    elif type == 'eq':
        image[image == value] = setto
    elif type == 'le':
        image[image >= value] = setto
    elif type == 'l':
        image[image > value] = setto

    return image


def binarize(image, value, type='l'):

    if type == 's':
        returnimg = (image < value)
    elif type == 'se':
        returnimg = (image <= value)
    elif type == 'eq':
        returnimg = (image == value)
    elif type == 'le':
        returnimg = (image >= value)
    elif type == 'l':
        returnimg = (image > value)

    return returnimg


def conncomp(image, neighborhood='direct', background_value=0):
    return vigra.analysis.labelMultiArrayWithBackground(image, neighborhood=neighborhood, background_value=background_value)


def crop(image, start, stop):
    return image[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]

# _____________________________________________________________________________________________


class ImageProcessing:

    _data = None

    def __init__(self, data, key=None):

        self.set_data(data, key=key)

        if data is None:
            print "ImageProcessing: Empty construction."

    @classmethod
    def empty(cls):
        return cls(None)

    ###########################################################################################
    # Data operations

    def set_data(self, data, key=None):
        if key is None:
            self._data = data
        else:
            self._data = {key, data}

    def set_data_dict(self, data, append=False):

        if not append or self._data is None:
            self._data = {}

        for d in data:
            self._data[d] = data[d]

    def get_data(self):
        return self._data

    def get_image(self, id=None):

        if id is None:
            if type(self._data) is dict:
                return None
            else:
                return self._data
        else:
            if type(self._data) is dict:
                return self._data[id]
            else:
                return None

    def converttodict(self, name):
        if type(self._data) is dict:
            print 'Warning: Type already a dict!'
            return
        else:
            t = self._data
            self._data = {}
            self._data[name] = t

    def addtodict(self, data, name):
        if type(self._data) is dict:
            self._data[name] = data
        else:
            print 'Warning: Not like this, convert to dict first!'

    def deepcopy_entry(self, sourcekey, targetkey):
        if type(self._data) is dict:
            # self._data[targetkey] = copy.deepcopy(self._data[sourcekey])
            self._data[targetkey] = self._data[sourcekey]
        else:
            print 'Warning: Deepcopy only implemented for dict type!'

    ###########################################################################################
    # Image processing

    def anytask(self, function, ids, *args, **kwargs):
        if type(self._data) is dict:
            if ids is None:
                for id in self._data:
                    self._data[id] = function(self._data[id], *args, **kwargs)
            else:
                for id in ids:
                    self._data[id] = function(self._data[id], *args, **kwargs)
        else:
            self._data = function(self._data, *args, **kwargs)

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
        if type(self._data) is dict:
            returndict = {}
            if ids is None:
                for id in self._data:
                    returndict[id] = function(self._data[id], *args, **kwargs)
            else:
                for id in ids:
                    returndict[id] = function(self._data[id], *args, **kwargs)
            return returndict
        else:
            return function(self._data, *args, **kwargs)

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

    def astype(self, dtype, ids=None):
        self.anytask(astype, ids, dtype)

    def distance_transform(self, pixel_pitch=(), ids=None):
        self.anytask(distance_transform, ids, pixel_pitch=pixel_pitch)

    def filter_values(self, value, type='se', setto=0, ids=None):
        self.anytask(filter_values, ids, value, type=type, setto=setto)

    def binarize(self, value, type='l', ids=None):
        self.anytask(binarize, ids, value, type=type)

    def conncomp(self, neighborhood='direct', background_value=0, ids=None):
        self.anytask(conncomp, ids, neighborhood=neighborhood, background_value=background_value)

    def skimage_watershed(self, markers, connectivity=1, offset=None, mask=None,
                          compactness=0, ids=None):
        self.anytask(watershed, ids, markers, connectivity=connectivity,
                     offset=offset, mask=mask, compactness=compactness)

    def crop(self, start, stop, ids=None):
        self.anytask(crop, ids, start, stop)

    ###########################################################################################
    # Iterators

    def label_iterator(self, id=None):

        for lbl in np.unique(self.get_image(id)):
            yield lbl

    def label_image_iterator(self, from_id, to_id):

        for lbl in self.label_iterator(id=from_id):
            self.deepcopy_entry(from_id, to_id)
            self.getlabel(lbl, (to_id,))
            yield lbl
# _____________________________________________________________________________________________


class ImageFileProcessing(ImageProcessing):

    _imagePath = None
    _imageFile = None
    _imageFileName = None
    _imageNames = None
    _imageIds = 0
    # _data = None
    _boundaries = None

    def __init__(self, image_path, image_file, image_names=None, image_ids=None,
                 asdict=True, keys=None):

        if image_path is not None and image_file is not None:
            data = self.load_h5(image_path + image_file, image_ids, image_names,
                                asdict, keys)
        else:
            data = None

        ImageProcessing.__init__(self, data)

        self.set_file(image_path, image_file, image_names, image_ids)

    def load_h5(self, im_file, ids=None, names=None, asdict=False, keys=None):
        #     """
        #     :type str
        #     :param im_file: Full path to h5 file
        #
        #     :type list<int> or int
        #     :param im_ids: ID(s) of images to load
        #
        #     :type list<str> or str
        #     :param im_names: Name(s) of images to load
        #
        #     :type bool
        #     :param asdict: Load images as dictionary (True)
        #
        #     :type: list<str>
        #     :param keys: Dictionary keys, the names within the h5 file are used if not specified
        #
        #     :type ImageProcessing
        #     :return: Image processing object containing the h5 image contents
        #
        #     EXAMPLES
        #
        #     load_h5(self, '~/location/sample.h5',
        #             im_ids=0, im_names=None, asdict=False, keys=None)
        #     -> Loads one image in sample.h5
        #
        #     load_h5(self, '~/location/sample.h5',
        #             im_ids=None, im_names=None, asdict=True, keys=None)
        #     -> Loads all images in sample.h5 as dictionary
        #
        #     load_h5(self, '~/location/sample.h5',
        #             im_ids=None, im_names=('a', 'b', 'c'), asdict=True, keys=('bar', 'foo', 'baz'))
        #     -> Loads three images a, b, and c in sample.h5, assigning the keys bar, foo, and baz, respectively
        #     """

        # im_file must not be None!
        if im_file is None:
            return None

        # Open the stream
        f = h5py.File(im_file)

        # Make sure im_names is not None
        # Return if this is not possible
        if names is None:
            names = f.keys()
            if ids is not None:
                names = [names[x] for x in ids]
            if names is None:
                return None

        # Take the first entry if we do not want a dictionary
        if not asdict and type(names) is list:
            names = names[0]

        # print names

        # Make sure keys is not None
        if keys is None:
            keys = names

        if asdict:
            # Initialize as dict ...

            images = {}

            i = 0
            for n in names:

                images[keys[i]] = np.array(f.get(n))
                i += 1

            return images

        else:

            return np.array(f.get(names))

    @classmethod
    def empty(cls):
        """ Empty construction is intended for debugging purposes """
        return cls(None, None)

    def set_file(self, image_path, image_file, image_names, image_ids):
        self._imagePath = image_path
        self._imageFile = image_file
        self._imageNames = image_names
        self._imageIds = image_ids
        if self._imageFile is not None:
            self._imageFileName = re.sub('\.h5$', '', self._imageFile)

    def get_filename(self):
        return self._imageFileName + '.h5'

    def addtoname(self, addstr):
        self._imageFileName += addstr

    ###########################################################################################
    # Log file operations

    _logger = None

    def startlogger(self, filename=None, type='a'):

        if filename is not None:
            self._logger = open(filename, type)

        self.logging("Logger started: {}\n".format(time.strftime('%X %z on %b %d, %Y')))

    def logging(self, format, *args):
        print format.format(*args)
        format += "\n"
        if self._logger is not None:
            self._logger.write(format.format(*args))

    def stoplogger(self):

        self.logging("Logger stopped: {}".format(time.strftime('%X %z on %b %d, %Y')))

        if self._logger is not None:
            self._logger.close()

    ###########################################################################################
    # Image processing

    def anytask_fp(self, function, addtofilename, ids, *args, **kwargs):
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
        ImageProcessing.anytask(self, function, ids, *args, **kwargs)
        # self._data.anytask(function, ids, *args, **kwargs)
        if addtofilename is not None:
            self._imageFileName += addtofilename
        else:
            self._imageFileName += '.modified'

    def invert_image(self, ids=None):
        ImageProcessing.invert_image(self, ids=ids)
        self._imageFileName += '.inv'

    def swapaxes(self, axis1, axis2, ids=None):
        ImageProcessing.swapaxes(self, axis1, axis2, ids=ids)
        self._imageFileName += '.swpxs_' + str(axis1) + '_' + str(axis2)

    def rollaxis(self, axis, start=0, ids=None):
        ImageProcessing.rollaxis(self, axis, start, ids=ids)
        self._imageFileName += '.rllxs_' + str(axis) + '_' + str(start)

    def resize(self, zoom, mode, ids=None):
        ImageProcessing.resize(self, zoom, mode, ids=ids)
        self._imageFileName += '.resize'

    def resize_z_nearest(self, z, ids=None):
        ImageProcessing.resize_z_nearest(self, z, ids=ids)
        self._imageFileName += '.resizez_' + str(z)

    def getlabel(self, label, ids=None):
        ImageProcessing.getlabel(self, label, ids=ids)
        self._imageFileName += '.lbl_' + str(label)

    def distance_transform(self, pixel_pitch=(), ids=None):
        ImageProcessing.distance_transform(self, pixel_pitch=pixel_pitch, ids=ids)
        self._imageFileName += '.dt'

    def filter_values(self, value, type='se', setto=0, ids=None):
        ImageProcessing.filter_values(self, value, type=type, setto=setto, ids=ids)
        self._imageFileName += '.filt_{}_{}'.format(type, value)

    def binarize(self, value, type='l', ids=None):
        ImageProcessing.binarize(self, value, type=type, ids=ids)
        self._imageFileName += '.bin_{}_{}'.format(type, value)

    def conncomp(self, neighborhood='direct', background_value=0, ids=None):
        ImageProcessing.conncomp(self, neighborhood=neighborhood, background_value=background_value, ids=ids)
        self._imageFileName += '.conncomp'

    def skimage_watershed(self, markers, connectivity=1, offset=None, mask=None,
                          compactness=0, ids=None):
        ImageProcessing.skimage_watershed(self, markers, connectivity=connectivity, offset=offset,
                                     mask=mask, compactness=compactness, ids=ids)
        self._imageFileName += '.ws'

    def crop(self, start, stop, ids=None):
        ImageProcessing.crop(self, start, stop, ids=ids)
        self._imageFileName += '.crop_{}-{}-{}_{}-{}-{}'.format(start[0], start[1], start[2], stop[0], stop[1], stop[2])

    ###########################################################################################
    # Write h5 files

    def write_h5(self, nfile, data, image_names=None, dict_ids=None):
        # print "Writing..."
        of = h5py.File(self._imagePath + nfile)

        self.logging("Writing to file: {}".format(self._imagePath + nfile))

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

    def write(self, ids=None, filename=None):
        """
        :type list
        :param ids: Set only if data is a dict!
            Use to specify which entries in the data dictionary are written to file
            Default: None (everything is written)
        """
        if filename is None:
            self.write_h5(self._imageFileName + '.h5', self.get_data(), dict_ids=ids)
        else:
            self.write_h5(filename, self.get_data(), dict_ids=ids)

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
    print ifp.get_data().shape
    print ifp.get_filename()

    # # Write the result (File name is automatically generated depending on the performed operations)
    # ifp.write()


