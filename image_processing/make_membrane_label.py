
import h5py
import numpy as np
import re
from scipy import ndimage
import random
# import scipy

__author__ = 'jhennies'


class ImageFileProcessing:

    _imagePath = ''
    _imageFile = ''
    _imageFileName = ''
    _imageName = None
    _imageID = 0
    _data = None
    _boundaries = None

    def __init__(self):
        pass

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
            self._data = np.array(f.get(self._imageName))
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

    def write_h5(self, nfile, data, image_name=None):
        print "Writing..."
        of = h5py.File(self._imagePath + nfile)
        if image_name is None:
            of.create_dataset(self._imageName, data=data)
        else:
            of.create_dataset(image_name, data=data)
        of.close()

    def crop_h5(self, crop):
        # INPUT
        #   crop: Tuple (size_d1, size_d2, ... , size_dn)

        # Load image
        self.load_h5()

        # Crop image
        cropped_data = self._data[0:crop[0], 0:crop[1], 0:crop[2]]

        # Write cropped image
        cropped_file = self._imageFileName + '.crop_' + str(crop[0]) + '_' + str(crop[1]) + '_' + str(crop[2]) + '.h5'
        self.write_h5(cropped_file, cropped_data)

    def stack_h5(self, target_file_name, stack_file, stack_id=None, stack_name=None):

        if self._data is None:
            self.load_h5()

        print stack_file
        data = self.load_h5(im_file=stack_file, im_id=stack_id, im_name=stack_name)
        print data.shape
        print self._data.shape
        stack = np.array([self._data, data])
        print stack.shape
        self.write_h5(target_file_name, stack)

    def mark_boundaries(self, boundary_label=None, ignore_label=None, ignore_width=0, name=None):

        if self._data is None:
            self.load_h5()

        # Add image-offsets #################
        # Avoid division by zero!
        self._data += 1
        # Increase data size by one pixel in each relevant direction
        bdata = np.ones((self._data.shape[0] + 2, self._data.shape[1] + 2, self._data.shape[2]), dtype=np.float)
        bdata[1:-1, 1:-1, :] = self._data
        bdata[1:-1, 0, :] = self._data[:, 0, :]
        bdata[1:-1, -1, :] = self._data[:, -1, :]
        bdata[0, 1:-1, :] = self._data[0, :, :]
        bdata[-1, 1:-1, :] = self._data[-1, :, :]
        # print bdata[0:10, 0:10, 50]
        # print bdata[-10:, -10:, 50]

        print bdata.shape
        print self._data.shape
        # print self._data[1:-1, 1:-1, 1:-1].shape
        self._boundaries = bdata[0:-2, 1:-1, :] + bdata[2:, 1:-1, :]
        self._boundaries = self._boundaries + bdata[1:-1, 0:-2, :] + bdata[1:-1, 2:, :]
        # self._boundaries = self._boundaries + self._data[1:-1, 1:-1, 0:-2] + self._data[1:-1, 1:-1, 2:]
        self._boundaries /= bdata[1:-1, 1:-1, :]
        self._boundaries[self._boundaries < 4] = 1
        self._boundaries[self._boundaries > 4] = 1
        self._boundaries[self._boundaries == 4] = 0
        # print self._boundaries[0:10, 0:10, 20]
        cbdata = bdata[1:-1, 1:-1, :]
        if boundary_label is not None:
            self._boundaries[cbdata == boundary_label] = 1
        # print self._boundaries[0:10, 0:10, 20]

        # Add insecurity offset
        if ignore_width > 0:
            dil_boundaries = ndimage.binary_dilation(
                self._boundaries, structure=np.ones((3, 3, 1)), iterations=ignore_width)
        else:
            dil_boundaries = self._boundaries
        dil_boundaries = dil_boundaries - self._boundaries
        # print dil_boundaries[0:10, 0:10, 20]

        # Set the final boundary label
        self._boundaries[self._boundaries == 1] = 2
        # Set the final background label
        self._boundaries[self._boundaries == 0] = 1
        # Add the ignore width
        self._boundaries[dil_boundaries == 1] = 0
        # print self._boundaries[0:10, 0:10, 20]

        # Ignore the ignore label
        if ignore_label is not None:
            ign = np.zeros(self._data.shape)
            ign[self._data == ignore_label] = 1
            ign[self._boundaries == 2] = 0
            self._boundaries[ign == 1] = 0

        self._boundaries = self._boundaries.astype(np.int32)
        # print self._boundaries.dtype

        if name is None:
            self.write_h5('boundaries.h5', self._boundaries)
        else:
            self.write_h5(name, self._boundaries)

    def split_data_h5(self, file_name='split_data.h5', split_dimension=0, squeeze=False):
        # Load data
        self.load_h5()

        exec_code = 'self._data[' + ':, ' * split_dimension + 'i' + ', :' * (self._data.ndim - split_dimension - 1) + ']'
        print exec_code
        # new_data = [None for x in range(self._data.shape[split_dimension])]
        print self._data.shape

        for i in range(0, self._data.shape[split_dimension]):
            print i
            new_data = None
            exec 'new_data = ' + exec_code
            if squeeze:
                new_data = np.squeeze(new_data)
            print new_data.shape

            self.write_h5(file_name, new_data, image_name='data'+str(i))

    def count_labels(self, label=0):
        if self._data is None:
            self.load_h5()

        coords = np.nonzero(self._data == label)
        return len(coords[0])

    def randomly_convert_labels_h5(self, from_label=1, to_label=0, n=0, file_name=None, dataset_name='labels'):
        if self._data is None:
            self.load_h5()

        print self._data.shape
        # Get list of positions for the desired label
        coords = np.nonzero(self._data == from_label)

        n_total = len(coords[0])
        print n_total

        if n >= n_total:
            print 'n > n_total'
            return None

        # Randomly select n coordinates for conversion
        new_coords = ()
        rnd = random.sample(xrange(n_total), n)
        for i in range(0, len(coords)):
            new_coords += (coords[i][rnd],)
            print i

        self._data[new_coords] = to_label

        if file_name is not None:
            print file_name
            self.write_h5(file_name, self._data, image_name=dataset_name)
        else:
            return self._data


if __name__ == "__main__":
    imagePath = '/windows/mobi/h1.hci/isbi_2013/data/'
    imageFile = 'ground-truth.h5'
    imageName = None
    imageID = 0

    # Initialize ImageFileProcessing object
    ifp = ImageFileProcessing()
    ifp.set_file(imagePath, imageFile, imageName, imageID)

    # Crop image if necessary, then re-run with cropped image
    # ifp.crop_h5((100, 100, 100))
    # quit()

    # Large membrane areas (label == 1) such as myelin sheets are not labeled
    # ifp.mark_boundaries(boundary_label=None, ignore_width=4, ignore_label=1, name='boundaries.blNone.ign4.ignl1.h5')

    # Large membrane areas are labeled as membrane
    ifp.mark_boundaries(boundary_label=1, ignore_width=4, ignore_label=None, name='boundaries.bl1.ign4.h5')



