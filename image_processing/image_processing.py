
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
    return np.array((image == label)).astype(np.uint32)


def amax(image):
    return np.amax(image)


def astype(image, dtype):
    return image.astype(dtype=dtype)


def distance_transform(image, pixel_pitch=(), background=True):
    return vigra.filters.distanceTransform(image, pixel_pitch=pixel_pitch, background=background)


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

    returnimg = np.array()
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


def shape(image):
    return image.shape


def power(image, value):
    return np.power(image, value)


def mult(image, value):
    return np.multiply(image, value)


def mult2im(image1, image2):
    return image1 * image2


def add(image, value):
    return image + value


def add2im(image1, image2):
    return image1 + image2


def concatenate(image1, image2):
    return [image1, image2]


def find_bounding_rect(image):

    # Bands
    bnds = np.flatnonzero(image.sum(axis=0).sum(axis=0))
    # Rows
    rows = np.flatnonzero(image.sum(axis=1).sum(axis=1))
    # Columns
    cols = np.flatnonzero(image.sum(axis=2).sum(axis=0))

    return [[rows.min(), rows.max()+1], [cols.min(), cols.max()+1], [bnds.min(), bnds.max()+1]]


def crop_bounding_rect(image, bounds=None):

    if bounds is None:
        bounds = find_bounding_rect(image)

    return image[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1], bounds[2][0]:bounds[2][1]]


def replace_subimage(image, rplimage, position=None, bounds=None, ignore=None):

    # Make sure bounds is not None
    if bounds is None and position is None:
        bounds = ((0, 0), (0, 0), (0, 0))
    if bounds is None and position is not None:
        bounds = ((position[0], rplimage.shape[0]), (position[1], rplimage.shape[1]), (position[2], rplimage.shape[2]))

    if ignore is None:
        image[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1], bounds[2][0]:bounds[2][1]] = rplimage
    else:
        image[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1], bounds[2][0]:bounds[2][1]][rplimage != ignore] = rplimage[rplimage != ignore]

    return image


def mask_image(image, mask, maskvalue=False, value=0):
    image[mask == maskvalue] = value
    return image

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
            self._data = {key: data}

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
            self._data[targetkey] = copy.deepcopy(self._data[sourcekey])
            # self._data[targetkey] = self._data[sourcekey]
        else:
            print 'Warning: Deepcopy only implemented for dict type!'

    ###########################################################################################
    # Image processing

    def anytask(self, task, *args, **kwargs):
        """
        :param task: The function which will be executed (has to take an image as first argument or first two arguments)
        :param ids=None: The images (respective ids) which will be used for calculation
        :param targetids=None: Target image (respective id) where the result will be stored
        :param ids2=None: Images (respective ids) which will be passed as the second argument to task
        :param args: Remaining arguments of task
        :param kwargs: Remaining keyword arguments of task

        EXAMPLES:

        # Run the function mult which accepts one image and a value as arguments on the images with ids 'key1' and
        # 'key2' and write them to the dictionary entries 'target1' and 'target2', respectively.
        def mult(image, value):
            return image * value
        anytask(mult, 10, ids=('key1', 'key2'), targetids=('target1', 'target2'))

        # Run the function add2ims which accepts two images as arguments on the images with keys 'key1' and 'key2' and
        # write the result to the dictionary entry 'target'.
        def add2ims(image1, image2):
            return image1 + image2
        anytask(add2ims, ids='key1', ids2='key2', targetids='target')

        """

        if type(self._data) is dict:

            # Defaults
            ids = None
            ids2 = None
            targetids = None

            # Get keyword arguments
            if 'ids' in kwargs.keys():
                ids = kwargs.pop('ids')
                if type(ids) is str:
                    ids = (ids,)

            if ids is None:
                ids = self._data.keys()

            if 'ids2' in kwargs.keys():
                ids2 = kwargs.pop('ids2')
                if type(ids2) is str:
                    ids2 = (ids2,)

            # ids and ids2 have to have the same length
            if ids2 is not None:
                if len(ids) != len(ids2):
                    if len(ids2) == 1:
                        ids2 *= len(ids)
                    else:
                        raise NameError('Error in ImageProcessing.anytask: len(ids2) is not equal len(ids) or 1.')

            if 'targetids' in kwargs.keys():
                targetids = kwargs.pop('targetids')
                if targetids is not None:
                    if type(targetids) is str:
                        targetids = (targetids,)
                    if len(targetids) != len(ids):
                        # print 'Warning: len(targetids) is not equal to len(ids)! Nothing was computed.'
                        raise NameError('Error in ImageProcessing.anytask: len(targetids) is not equal to len(ids)!')

            if targetids is None:
                targetids = ids

            if ids2 is None:
                allids = zip(ids, targetids)
            else:
                allids = zip(ids, ids2, targetids)

            for allid in allids:
                if len(allid) == 2:
                    if allid[1] != allid[0]:
                        self.deepcopy_entry(allid[0], allid[1])
                        self._data[allid[1]] = task(self._data[allid[1]], *args, **kwargs)
                    else:
                        self._data[allid[1]] = task(self._data[allid[0]], *args, **kwargs)
                else:
                    if allid[2] != allid[0]:
                        self.deepcopy_entry(allid[0], allid[2])
                        self._data[allid[2]] = task(self._data[allid[2]], self._data[allid[1]], *args, **kwargs)
                    else:
                        self._data[allid[2]] = task(self._data[allid[0]], self._data[allid[1]], *args, **kwargs)

        else:

            self._data = task(self._data, *args, **kwargs)

    def anytask_rtrn(self, task, *args, **kwargs):
        """
        Works like anytask, with the difference that a value will be returned instead of written to the data dictionary
        of this class.

        :type function
        :param task:
            A function in the form: func(image, *args, **kwargs)
            Note that the first parameter is fixed to the currently loaded image

        :type list
        :param ids=None:
            list of keys denoting the dictionary entries in self._data._image which will be processed
            Set to None if self._data is not a dictionary or for processing of all entries
        """

        if type(self._data) is dict:

            # Defaults
            ids = None
            ids2 = None

            # Get keyword arguments
            if 'ids' in kwargs.keys():
                ids = kwargs.pop('ids')
                if type(ids) is str:
                    ids = (ids,)

            if ids is None:
                ids = self._data.keys()

            if 'ids2' in kwargs.keys():
                ids2 = kwargs.pop('ids2')
                if type(ids2) is str:
                    ids2 = (ids2,)

            if ids2 is None:
                allids = zip(ids,)
            else:
                allids = zip(ids, ids2)

            returndict = {}
            for allid in allids:
                if len(allid) == 1:
                    returndict[allid[0]] = task(self._data[allid[0]], *args, **kwargs)
                else:
                    returndict[allid[0]] = task(self._data[allid[0]], self._data[allid[1]], *args, **kwargs)
            if len(returndict) == 1:
                return returndict[returndict.keys()[0]]
            else:
                return returndict

        else:

            return task(self._data, *args, **kwargs)

    def invert_image(self, ids=None, targetids=None):
        self.anytask(invert_image, ids=ids, targetids=targetids)

    def swapaxes(self, axis1, axis2, ids=None, targetids=None):
        self.anytask(swapaxes, axis1, axis2, ids=ids, targetids=targetids)

    def rollaxis(self, axis, start=0, ids=None, targetids=None):
        self.anytask(rollaxis, axis, start=start, ids=ids, targetids=targetids)

    def resize(self, zoom, mode, ids=None, targetids=None):
        self.anytask(resize, zoom, mode, ids=ids, targetids=targetids)

    def resize_z_nearest(self, z, ids=None, targetids=None):
        self.anytask(resize_z_nearest, z, ids=ids, targetids=targetids)

    def getlabel(self, label, ids=None, targetids=None):
        self.anytask(getlabel, label, ids=ids, targetids=targetids)

    def amax(self, ids=None):
        return self.anytask_rtrn(amax, ids=ids)

    def astype(self, dtype, ids=None, targetids=None):
        self.anytask(astype, dtype, ids=ids, targetids=targetids)

    def distance_transform(self, pixel_pitch=(), background=True, ids=None, targetids=None):
        self.anytask(distance_transform, pixel_pitch=pixel_pitch, background=background, ids=ids, targetids=targetids)

    def filter_values(self, value, type='se', setto=0, ids=None, targetids=None):
        self.anytask(filter_values, value, type=type, setto=setto, ids=ids, targetids=targetids)

    def binarize(self, value, type='l', ids=None, targetids=None):
        self.anytask(binarize, value, type=type, ids=ids, targetids=targetids)

    def conncomp(self, neighborhood='direct', background_value=0, ids=None, targetids=None):
        self.anytask(conncomp, neighborhood=neighborhood, background_value=background_value, ids=ids, targetids=targetids)

    def skimage_watershed(self, markers, connectivity=1, offset=None, mask=None,
                          compactness=0, ids=None, targetids=None):
        self.anytask(watershed, markers, connectivity=connectivity,
                     offset=offset, mask=mask, compactness=compactness, ids=ids, targetids=targetids)

    def crop(self, start, stop, ids=None, targetids=None):
        self.anytask(crop, start, stop, ids=ids, targetids=targetids)

    def shape(self, ids=None):
        return self.anytask_rtrn(shape, ids=ids)

    def power(self, value, ids=None, targetids=None):
        self.anytask(power, value, ids=ids, targetids=targetids)

    def mult(self, value, ids=None, targetids=None):
        self.anytask(mult, value, ids=ids, targetids=targetids)

    def mult2im(self, ids=None, ids2=None, targetids=None):
        self.anytask(mult2im, ids=ids, ids2=ids2, targetids=targetids)

    def add(self, value, ids=None, targetids=None):
        self.anytask(add, value, ids=ids, targetids=targetids)

    def add2im(self, ids=None, ids2=None, targetids=None):
        self.anytask(add2im, ids=ids, ids2=ids2, targetids=targetids)

    def concatenate(self, ids, ids2, targetids=None):
        self.anytask(concatenate, ids=ids, ids2=ids2, targetids=targetids)

    def find_bounding_rect(self, ids=None):
        return self.anytask_rtrn(find_bounding_rect, ids=ids)

    def crop_bounding_rect(self, bounds=None, ids=None, targetids=None):
        self.anytask(crop_bounding_rect, bounds=bounds, ids=ids, targetids=targetids)

    def replace_subimage(self, position=None, bounds=None, ignore=None, ids=None, ids2=None, targetids=None):
        self.anytask(replace_subimage, position=position, bounds=bounds, ignore=ignore, ids=ids, ids2=ids2, targetids=targetids)

    def mask_image(self, maskvalue=False, value=0, ids=None, ids2=None, targetids=None):
        self.anytask(mask_image, maskvalue=maskvalue, value=value, ids=ids, ids2=ids2, targetids=targetids)

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

    def label_bounds_iterator(self, labelid, targetid, ids=None, targetids=None,
                              maskvalue=0, value=np.nan, minsize=None):

        for lbl in self.label_image_iterator(labelid, targetid):

            bounds = self.find_bounding_rect(ids=targetid)

            # def checkforminsize(bounds, minsize):
            #
            #     for i in xrange(0,3):
            #         print minsize[i]
            #         if (bounds[i][1] - bounds[i][0]) < minsize[i]:
            #             t = bounds[i][0] - (minsize[i] - (bounds[i][1] - bounds[i][0])) / 2
            #             bounds[i][1] += (minsize[i] - (bounds[i][1] - bounds[i][0])) / 2
            #             bounds[i][0] = t
            #             if bounds[i][0] < 0:
            #                 bounds[i][1] -= bounds[i][0]
            #                 bounds[i][0] = 0
            #             if bounds[i][1] >= self.get_image(labelid).shape[i]:
            #                 bounds[i][0] -=
            #             print bounds[i]
            #
            #     return bounds

            # if minsize is not None:
            #     bounds = checkforminsize(bounds, minsize)

            self.crop_bounding_rect(bounds, ids=targetid)

            if ids is not None:
                self.crop_bounding_rect(bounds, ids=ids, targetids=targetids)
                self.mask_image(maskvalue=maskvalue, value=value, ids=targetids, ids2=targetid)

            yield {'bounds': bounds, 'label': lbl}

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

    def addfromfile(self, filename, image_ids=0, image_names=None, ids=None):
        if type(self._data) is dict:

            newdata = self.load_h5(filename, ids=image_ids, image_names=image_names, keys=ids)
            self.set_data_dict(newdata, append=True)

        else:
            ifp.logging('Warning: Convert to dict first, no file content was loaded.')
            return

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

    def anytask_fp(self, function, *args, **kwargs):
        """
        :type function
        :param function:
            A function in the form: func(image, *args, **kwargs)
            Note that the first parameter is fixed to the currently loaded image

        :type str
        :param addtofilename=None:
            Extension to the output file name; defaults to '.modified' when set to None
            If no extension is desired supply as empty string ('')

        :type list
        :param ids=None:
            list of keys denoting the dictionary entries in self._data._image which will be processed
            Set to None if self._data is not a dictionary or for processing of all entries
        """

        # Defaults
        addtofilename = None

        # Get keyword arguments
        if 'addtofilename' in kwargs.keys():
            addtofilename = kwargs.pop('addtofilename')

        ImageProcessing.anytask(self, function, *args, **kwargs)
        # self._data.anytask(function, ids, *args, **kwargs)
        if addtofilename is not None:
            self._imageFileName += addtofilename
        else:
            self._imageFileName += '.modified'

    def invert_image(self, ids=None, targetids=None):
        ImageProcessing.invert_image(self, ids=ids, targetids=targetids)
        self._imageFileName += '.inv'

    def swapaxes(self, axis1, axis2, ids=None, targetids=None):
        ImageProcessing.swapaxes(self, axis1, axis2, ids=ids, targetids=targetids)
        self._imageFileName += '.swpxs_' + str(axis1) + '_' + str(axis2)

    def rollaxis(self, axis, start=0, ids=None, targetids=None):
        ImageProcessing.rollaxis(self, axis, start, ids=ids, targetids=targetids)
        self._imageFileName += '.rllxs_' + str(axis) + '_' + str(start)

    def resize(self, zoom, mode, ids=None, targetids=None):
        ImageProcessing.resize(self, zoom, mode, ids=ids, targetids=targetids)
        self._imageFileName += '.resize'

    def resize_z_nearest(self, z, ids=None, targetids=None):
        ImageProcessing.resize_z_nearest(self, z, ids=ids, targetids=targetids)
        self._imageFileName += '.resizez_' + str(z)

    def getlabel(self, label, ids=None, targetids=None):
        ImageProcessing.getlabel(self, label, ids=ids, targetids=targetids)
        self._imageFileName += '.lbl_' + str(label)

    def distance_transform(self, pixel_pitch=(), background=True, ids=None, targetids=None):
        ImageProcessing.distance_transform(self, pixel_pitch=pixel_pitch, background=background, ids=ids, targetids=targetids)
        self._imageFileName += '.dt'

    def filter_values(self, value, type='se', setto=0, ids=None, targetids=None):
        ImageProcessing.filter_values(self, value, type=type, setto=setto, ids=ids, targetids=targetids)
        self._imageFileName += '.filt_{}_{}'.format(type, value)

    def binarize(self, value, type='l', ids=None, targetids=None):
        ImageProcessing.binarize(self, value, type=type, ids=ids, targetids=targetids)
        self._imageFileName += '.bin_{}_{}'.format(type, value)

    def conncomp(self, neighborhood='direct', background_value=0, ids=None, targetids=None):
        ImageProcessing.conncomp(self, neighborhood=neighborhood, background_value=background_value, ids=ids, targetids=targetids)
        self._imageFileName += '.conncomp'

    def skimage_watershed(self, markers, connectivity=1, offset=None, mask=None,
                          compactness=0, ids=None, targetids=None):
        ImageProcessing.skimage_watershed(self, markers, connectivity=connectivity, offset=offset,
                                     mask=mask, compactness=compactness, ids=ids, targetids=targetids)
        self._imageFileName += '.ws'

    def crop(self, start, stop, ids=None, targetids=None):
        ImageProcessing.crop(self, start, stop, ids=ids, targetids=targetids)
        self._imageFileName += '.crop_{}-{}-{}_{}-{}-{}'.format(start[0], start[1], start[2], stop[0], stop[1], stop[2])

    def power(self, value, ids=None, targetids=None):
        ImageProcessing.power(self, value, ids=ids, targetids=targetids)
        self._imageFileName += '.power_{}'.format(value)

    def mult(self, value, ids=None, targetids=None):
        ImageProcessing.mult(self, value, ids=ids, targetids=targetids)
        self._imageFileName += '.mult_{}'.format(value)

    def mult2im(self, ids=None, ids2=None, targetids=None):
        ImageProcessing.mult2im(self, ids=ids, ids2=ids2, targetids=targetids)
        self._imageFileName += '.mult2im'

    def add(self, value, ids=None, targetids=None):
        ImageProcessing.add(self, value, ids=ids, targetids=targetids)
        self._imageFileName += '.add_{}'.format(value)

    def add2im(self, ids=None, ids2=None, targetids=None):
        ImageProcessing.add2im(self, ids=ids, ids2=ids2, targetids=targetids)
        self._imageFileName += '.add2im'

    def concatenate(self, ids, ids2, targetids=None):
        ImageProcessing.concatenate(self, ids, ids2, targetids=targetids)
        self._imageFileName += '.conced'

    def crop_bounding_rect(self, bounds=None, ids=None, targetids=None):
        ImageProcessing.crop_bounding_rect(self, bounds=bounds, ids=ids, targetids=targetids)
        self._imageFileName += '.crop_bounds'

    def replace_subimage(self, position=None, bounds=None, ignore=None, ids=None, ids2=None, targetids=None):
        ImageProcessing.replace_subimage(self, position=position, bounds=bounds, ignore=ignore, ids=ids, ids2=ids2, targetids=targetids)
        self._imageFileName += '.rplsubim'

    def mask_image(self, maskvalue=False, value=0, ids=None, ids2=None, targetids=None):
        ImageProcessing.mask_image(self, maskvalue=maskvalue, value=value, ids=ids, ids2=ids2, targetids=targetids)
        self._imageFileName += '.masked'

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
                of.create_dataset(self._imageNames, data=data)
            else:
                of.create_dataset(image_names, data=data)

        of.close()

    def write(self, ids=None, filename=None):
        """
        :type ids: list
        :param ids: Set only if data is a dict!
            Use to specify which entries in the data dictionary are written to file
            Default: None (everything is written)

        :type filename: str
        :param filename: If not set the generated file name within this class is used
        """
        if filename is None:
            self.write_h5(self._imageFileName + '.h5', self.get_data(), dict_ids=ids)
        else:
            self.write_h5(filename, self.get_data(), dict_ids=ids)

# _____________________________________________________________________________________________


if __name__ == "__main__":

    ### EXAMPLE ###

    folder = '/media/julian/Daten/neuraldata/cremi_2016/'
    filename = 'cremi.splA.raw_neurons.crop.h5'
    names = ('neuron_ids',)
    keys = ('labels',)

    # Create object and define image file
    ifp = ImageFileProcessing(
        folder,
        filename, asdict=True,
        image_names=names,
        keys=keys)

    ifp.startlogger()

    # Modify the image(s)...

    # ... with some pre-defined operations
    ifp.invert_image()
    ifp.swapaxes(0, 2)
    ifp.rollaxis(1, 0)

    # ... with an arbitrary function
    def plusx(array, x):
        return array + x
    ifp.anytask_fp(plusx, 5, addtofilename='.plus5', ids='labels', targetids='labelsplus5')

    # Functions taking two images as argument
    ifp.concatenate(ids='labels', ids2='labelsplus5', targetids='concated')
    ifp.add2im(ids='labels', ids2='labelsplus5', targetids='added')
    ifp.mult2im(ids='labels', ids2='labelsplus5', targetids='multed')

    # Getter functions
    ifp.logging('ifp.get_data().keys() = {}', ifp.get_data().keys())
    ifp.logging('ifp.get_filename() = {}', ifp.get_filename())

    # Return functions (These type of functions do not change the involved image(s))
    ifp.logging('Maximum values = {}', ifp.amax())

    # # Write the result (File name is automatically generated depending on the performed operations)
    # ifp.write()

    ifp.stoplogger()


