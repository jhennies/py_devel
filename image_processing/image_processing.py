
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
import sys
import traceback
from simple_logger import SimpleLogger
from yaml_parameters import YamlParams
import processing_lib as lib
from hdf5_processing import Hdf5Processing

__author__ = 'jhennies'


class ImageProcessing(SimpleLogger):

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

    def rename_entries(self, ids, targetids):

        if type(ids) is str:
            ids = (ids,)
            if type(targetids) is not str:
                raise NameError("Error in ImageProcessing.rename_entries: ids and targetids not compatible!")
            targetids = (targetids,)

        if len(ids) != len(targetids):
            raise NameError("Error in ImageProcessing.rename_entries: ids and targetids do not have same length!")

        for i in xrange(0, len(ids)):
            self._data[targetids[i]] = self._data.pop(ids[i])

    def new_image(self, shape, name, dtype=np.float32, value=0):
        image = np.ones(shape, dtype=dtype) * value
        self.set_data_dict({name: image})

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
        self.anytask(lib.invert_image, ids=ids, targetids=targetids)

    def swapaxes(self, axis1, axis2, ids=None, targetids=None):
        self.anytask(lib.swapaxes, axis1, axis2, ids=ids, targetids=targetids)

    def rollaxis(self, axis, start=0, ids=None, targetids=None):
        self.anytask(lib.rollaxis, axis, start=start, ids=ids, targetids=targetids)

    def resize(self, zoom, mode, ids=None, targetids=None):
        self.anytask(lib.resize, zoom, mode, ids=ids, targetids=targetids)

    def resize_z_nearest(self, z, ids=None, targetids=None):
        self.anytask(lib.resize_z_nearest, z, ids=ids, targetids=targetids)

    def getlabel(self, label, ids=None, targetids=None):
        self.anytask(lib.getlabel, label, ids=ids, targetids=targetids)

    def amax(self, ids=None):
        return self.anytask_rtrn(lib.amax, ids=ids)

    def astype(self, dtype, ids=None, targetids=None):
        self.anytask(lib.astype, dtype, ids=ids, targetids=targetids)

    def distance_transform(self, pixel_pitch=(), background=True, ids=None, targetids=None):
        self.anytask(lib.distance_transform, pixel_pitch=pixel_pitch, background=background, ids=ids, targetids=targetids)

    def filter_values(self, value, type='se', setto=0, ids=None, targetids=None):
        self.anytask(lib.filter_values, value, type=type, setto=setto, ids=ids, targetids=targetids)

    def binarize(self, value, type='l', ids=None, targetids=None):
        self.anytask(lib.binarize, value, type=type, ids=ids, targetids=targetids)

    def conncomp(self, neighborhood='direct', background_value=0, ids=None, targetids=None):
        self.anytask(lib.conncomp, neighborhood=neighborhood, background_value=background_value, ids=ids, targetids=targetids)

    def skimage_watershed(self, markers, connectivity=1, offset=None, mask=None,
                          compactness=0, ids=None, targetids=None):
        self.anytask(watershed, markers, connectivity=connectivity,
                     offset=offset, mask=mask, compactness=compactness, ids=ids, targetids=targetids)

    def crop(self, start, stop, ids=None, targetids=None):
        self.anytask(lib.crop, start, stop, ids=ids, targetids=targetids)

    def shape(self, ids=None):
        return self.anytask_rtrn(lib.shape, ids=ids)

    def power(self, value, ids=None, targetids=None):
        self.anytask(lib.power, value, ids=ids, targetids=targetids)

    def mult(self, value, ids=None, targetids=None):
        self.anytask(lib.mult, value, ids=ids, targetids=targetids)

    def mult2im(self, ids=None, ids2=None, targetids=None):
        self.anytask(lib.mult2im, ids=ids, ids2=ids2, targetids=targetids)

    def add(self, value, ids=None, targetids=None):
        self.anytask(lib.add, value, ids=ids, targetids=targetids)

    def add2im(self, ids=None, ids2=None, targetids=None):
        self.anytask(lib.add2im, ids=ids, ids2=ids2, targetids=targetids)

    def concatenate(self, ids, ids2, targetids=None):
        self.anytask(lib.concatenate, ids=ids, ids2=ids2, targetids=targetids)

    def find_bounding_rect(self, ids=None):
        return self.anytask_rtrn(lib.find_bounding_rect, ids=ids)

    def crop_bounding_rect(self, bounds=None, ids=None, targetids=None):
        self.anytask(lib.crop_bounding_rect, bounds=bounds, ids=ids, targetids=targetids)

    def replace_subimage(self, position=None, bounds=None, ignore=None, ids=None, ids2=None, targetids=None):
        self.anytask(lib.replace_subimage, position=position, bounds=bounds, ignore=ignore, ids=ids, ids2=ids2, targetids=targetids)

    def mask_image(self, maskvalue=False, value=0, ids=None, ids2=None, targetids=None):
        self.anytask(lib.mask_image, maskvalue=maskvalue, value=value, ids=ids, ids2=ids2, targetids=targetids)

    def unique(self, ids=None):
        return self.anytask_rtrn(lib.unique, ids=ids)

    def gaussian_smoothing(self, sigma, ids=None, targetids=None):
        self.anytask(lib.gaussian_smoothing, sigma, ids=ids, targetids=targetids)

    def extended_local_maxima(self, neighborhood=26, ids=None, targetids=None):
        self.anytask(lib.extended_local_maxima, neighborhood=neighborhood, ids=ids, targetids=targetids)

    def pixels_at_boundary(self, axes=[1, 1, 1], ids=None, targetids=None):
        self.anytask(lib.pixels_at_boundary, axes=axes, ids=ids, targetids=targetids)

    def positions2value(self, coordinates, value, ids=None, targetids=None):
        self.anytask(lib.positions2value, coordinates, value, ids=ids, targetids=targetids)

    ###########################################################################################
    # Cross-computation of two ImageProcessing instances

    def cross_comp(self, improc, task, *args, **kwargs):

        if 'out' in kwargs.keys():
            hfp = kwargs.pop('out')
        else:
            hfp = Hdf5Processing()

        reverse_input = False
        if 'reverse_input' in kwargs.keys():
            reverse_input = kwargs.pop('reverse_input')

        data = self.get_data()
        if type(data) is not dict:
            data = {'data': data}

        external_data = improc.get_data()
        if type(external_data) is not dict:
            external_data = {'ext_data': data}

        for d_key, d_val in data.iteritems():

            for ex_key, ex_val in external_data.iteritems():

                if reverse_input:
                    hfp[d_key, ex_key] = task(ex_val, d_val, *args, **kwargs)
                else:
                    hfp[d_key, ex_key] = task(d_val, ex_val, *args, **kwargs)

        return hfp

    ###########################################################################################
    # Iterators

    def label_iterator(self, id=None, labellist=None, background=None):

        if labellist is None:
            labellist = self.unique(ids=id)

        if background is not None:
            labellist = filter(lambda x: x != 0, labellist)

        for lbl in labellist:
            yield lbl

    def label_image_iterator(self, from_id, to_id, labellist=None, background=None, accumulate=False):

        if accumulate:
            self.addtodict(np.zeros(self.shape(from_id)), to_id)

        for lbl in self.label_iterator(id=from_id, labellist=labellist, background=background):
            if not accumulate:
                self.getlabel(lbl, ids=from_id, targetids=to_id)
            else:
                newlabel = lib.getlabel(lbl, self.get_image(from_id))
                self.get_image(to_id)[newlabel == 1] = lbl

            yield lbl

    def label_bounds_iterator(self, labelid, targetid, ids=None, targetids=None,
                              maskvalue=0, value=np.nan, background=None, labellist=None,
                              forcecontinue=False):

        for lbl in self.label_image_iterator(labelid, targetid, background=background, labellist=labellist):

            # self.logging('self.amax() = {}', self.amax())
            try:

                bounds = self.find_bounding_rect(ids=targetid)

                self.crop_bounding_rect(bounds, ids=targetid)

                if ids is not None:
                    self.crop_bounding_rect(bounds, ids=ids, targetids=targetids)
                    self.mask_image(maskvalue=maskvalue, value=value, ids=targetids, ids2=targetid)

                yield {'bounds': bounds, 'label': lbl}

            except:

                if forcecontinue:
                    self.errprint('Warning: Something went wrong in label {}, jumping to next label'.format(lbl), traceback)

                    continue
                else:
                    raise

    def labelpair_bounds_iterator(self, labelid, targetid, ids=None, targetids=None,
                                  maskvalue=0, value=np.nan, labellist=None,
                                  forcecontinue=False):

        if labellist is None:
            raise ValueError('Error in ImageProcessing.labelpair_bounds_iterator: A labellist with labelpairs has to be specified!')

        for lblpair in labellist:

            try:
                self.getlabel(tuple(lblpair), ids=labelid, targetids=targetid)
                bounds = self.find_bounding_rect(ids=targetid)
                self.crop_bounding_rect(bounds, ids=targetid)

                if ids is not None:
                    self.crop_bounding_rect(bounds, ids=ids, targetids=targetids)
                    self.mask_image(maskvalue=maskvalue, value=value, ids=targetids, ids2=targetid)

                yield {'bounds': bounds, 'labels': lblpair}

            except:

                if forcecontinue:
                    self.errprint('Warning: Something went wrong in labelpair {}, jumping to next labelpair'.format(lblpair))
                    continue
                else:
                    raise

# _____________________________________________________________________________________________


class ImageFileProcessing(ImageProcessing, YamlParams):

    _imagePath = None
    _imageFile = None
    _imageFileName = None
    _imageNames = None
    _imageIds = 0
    # _data = None
    _boundaries = None

    def __init__(self, image_path=None, image_file=None, image_names=None, image_ids=None,
                 asdict=True, keys=None, yaml=None, yamlspec=None):
        """
        :param image_path:
        :param image_file:
        :param image_names:
        :param image_ids:
        :param asdict:
        :param keys:

        :type yaml: str
        :param yaml: Filename of yaml configuration file
            Can contain the fields image_path, image_file, image_names, image_ids, asdict, or keys
            If other names for the respective fields are desired use yamlspec (see below)

        :type yamlspec: dict
        :param yamlspec: Translates field names in the configuration file to the respective variable names

            EXAMPLES:

            When using a yaml file like this:
            ---YAML---
            image_path: '~/data/'
            image_file: 'mydatafile.h5'
            ----------
            yamlspec can be set to None.

            If other keywords are desired within the yaml file yamlspec has to be set accordingly:
            ---YAML---
            datapath: '~/data/'
            datafile: 'mydatafile.h5'
            ----------
            yamlspec={'image_path': 'datapath', 'image_file': 'datafile'}

            Note that not every variable has to be set within the yaml file, it is then taken from the function
            arguments.

            If yamlspec is set, only the variables specified in yamlspec are actually read:
            ---YAML---
            datapath: '~/data/'
            datafile: 'mydatafile.h5'
            asdict: True
            ----------
            yamlspec={'image_path': 'datapath', 'image_file': 'datafile'}
            --> asdict will be ignored and derived from the function arguments
            solution:
            yamlspec={'image_path': 'datapath', 'image_file': 'datafile', 'asdict': 'asdict'}

        """

        if yaml is not None:
            YamlParams.__init__(self, filename=yaml)
            # self._yaml = yaml
            # yamldict = self.load_yaml(yaml)
            yamldict = self.get_params()
            if yamlspec is None:
                if 'image_path' in yamldict.keys(): image_path = yamldict['image_path']
                if 'image_file' in yamldict.keys(): image_file = yamldict['image_file']
                if 'image_names' in yamldict.keys(): image_names = yamldict['image_names']
                if 'image_ids' in yamldict.keys(): image_ids = yamldict['image_ids']
                if 'asdict' in yamldict.keys(): asdict = yamldict['asdict']
                if 'keys' in yamldict.keys(): keys = yamldict['keys']

            else:
                if 'image_path' in yamlspec.keys(): image_path = yamldict[yamlspec['image_path']]
                if 'image_file' in yamlspec.keys(): image_file = yamldict[yamlspec['image_file']]
                if 'image_names' in yamlspec.keys():
                    if type(yamlspec['image_names']) is tuple:
                        image_names = ()
                        for i in xrange(1, len(yamlspec['image_names'])):
                            image_names += (yamldict[yamlspec['image_names'][0]][yamlspec['image_names'][i]],)
                    else:
                        image_names = yamldict[yamlspec['image_names']]
                if 'image_ids' in yamlspec.keys(): image_ids = yamldict[yamlspec['image_ids']]
                if 'asdict' in yamlspec.keys(): asdict = yamldict[yamlspec['asdict']]
                if 'keys' in yamlspec.keys(): keys = yamldict[yamlspec['keys']]
        else:
            YamlParams.__init__(self)

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

        # Make sure names is a list
        if asdict:
            if type(names) is str:
                names = (names,)

        # Take the first entry if we do not want a dictionary
        if not asdict and type(names) is list:
            names = names[0]

        # print names

        # Make sure keys is not None
        if keys is None:
            keys = names
        # Make sure keys is a list
        if type(keys) is str:
            keys = (keys,)

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

            newdata = self.load_h5(filename, ids=image_ids, names=image_names, keys=ids, asdict=True)
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

    def gaussian_smoothing(self, sigma, ids=None, targetids=None):
        ImageProcessing.gaussian_smoothing(self, sigma, ids=ids, targetids=targetids)
        self._imageFileName += '.gaussian_{}'.format(sigma)

    def extended_local_maxima(self, neighborhood=26, ids=None, targetids=None):
        ImageProcessing.extended_local_maxima(self, neighborhood=neighborhood, ids=ids, targetids=targetids)
        self._imageFileName += '.locmax'

    ###########################################################################################
    # Write h5 files

    def write_h5(self, data, nfile=None, filepath=None, image_names=None, dict_ids=None):
        # print "Writing..."

        if filepath is None and nfile is not None:
            of = h5py.File(self._imagePath + nfile)
            self.logging("Writing to file: {}".format(self._imagePath + nfile))
        elif filepath is not None:
            of = h5py.File(filepath)
            self.logging("Writing to file: {}".format(filepath))
        else:
            raise NameError("Error in ImageFileProcessing.write_h5: File name missing!")

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

    def write(self, ids=None, filename=None, filepath=None):
        """
        :type ids: list
        :param ids: Set only if data is a dict!
            Use to specify which entries in the data dictionary are written to file
            Default: None (everything is written)

        :type filename: str
        :param filename: If not set the generated file name within this class is used
        """

        if type(ids) is str:
            ids = (ids,)

        if filename is None and filepath is None:
            self.write_h5(self.get_data(), nfile=self._imageFileName + '.h5', dict_ids=ids)
        else:
            self.write_h5(self.get_data(), nfile=filename, filepath=filepath, dict_ids=ids)

        # elif filename is not None:
        #     self.write_h5(filename, self.get_data(), dict_ids=ids)
        #
        #
        # if filename is None:
        #     self.write_h5(self._imageFileName + '.h5', self.get_data(), dict_ids=ids)
        # else:
        #     self.write_h5(filename, self.get_data(), dict_ids=ids)


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


