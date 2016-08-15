
# from nasim, netdatautils import fromh5, toh5, slidingwindowslices
import netdatautils as nda
from netdatakit import cargo, feederweave
# import scipy
import time
import copy
import pickle
import math
import cPickle as pkl
import numpy as np

__author__ = 'jhennies'


class nn_upscale:

    _path = ""
    _datapath = ""
    _resultfile = ""
    _currentslice=None
    _datasetsize=None
    _nhoodsize=None
    _stride=None

    def __init__(self):
        pass

    def __init__(self, path=None, datapath=None,
                 resultfile=None,
                 datasetsize=[256, 256, 256],
                 nhoodsize=[32, 32, 32],
                 stride=16):
        self._path = path
        self._datapath = datapath
        self._datasetsize = datasetsize
        self._nhoodsize = nhoodsize
        self._stride = stride
        self._resultfile = resultfile

    def train_nn(self, roispath, popcubes=0, slicedimensions=[0, 1, 2]):
        """
        Creates feederweave object to train the neuronal network

        :type roispath: str
        :param roispath: Path to pickle file containing the regions of interest in the format:
            tuple( tuple(slice(x_start, x_end), slice(y_start, y_end), slice(z_start, z_end)), ... )

        :type popcubes: int
        :param popcubes: Number of cargo objects which are grouped together into one feederweave

        :type slicedimensions: list of int
        :param slicedimensions: Dimensions which will be used to create slices.
            When using [0, 1, 2] a cargo object for all three dimensions will be created, respectively

        :return: a feederweave containing a list of feederweaves each containing a list of cargo objects
        """

        # Load ROIs
        rois = self.open_rois(input=roispath)
        # Convert from tuple to list to make it poppable
        rois = list(rois)

        if popcubes == 0:
            popcubes = len(rois)

        fw = []

        # This loop is performed for a group of ROIs (amount = popcubes)
        while len(rois) > 0:

            # Randomly pop a list of slices
            data_slices = []
            for i in xrange(0, popcubes):
                # print i
                # print len(rois)
                data_slices += [rois.pop(np.random.randint(0, len(rois)))]
                if len(rois) == 0: break

            # Generate a list of cargo objects from these selected slices
            cg = self.make_cargo_list(data_slices, slicedimensions=slicedimensions)

            # Put this list into a feederweave object and append the list of feederweaves
            fw += [feederweave(cg)]

            print '_____________________________________________'
            print 'Remaining ROIs:'
            print len(rois)

        # One feederweave to represent them all...
        ffw = feederweave(fw)

        # It's done
        return ffw

    def get_slicedimension_axistag(self, dim):
        if dim == 0:
            return 'ijk'
        elif dim == 1:
            return 'jik'
        elif dim == 2:
            return 'kij'

    def make_cargo_list(self, data_slices, slicedimensions=[0, 1, 2]):

        cg = []

        for sl in data_slices:
            for dim in slicedimensions:
                # print sl
                axistag = self.get_slicedimension_axistag(dim)

                cg += [cargo(h5path=self._path, pathh5=self._datapath, data=None, axistags=axistag,
                            batchsize=1, nhoodsize=None, ds=None,
                            window=None, stride=None, preload=False, dataslice=sl,
                            preptrain=None, shuffleiterator=True)]

        return cg

    def load_data(self, currentslice):
        return nda.fromh5(self._path, datapath=self._datapath,
                      dataslice=currentslice, asnumpy=True, preptrain=None)

    def nn_on_cube_dimension(self, cube, dimension=0):

        cubeshape = cube.shape
        # Roll dimension such that the desired stack plane becomes the first
        cube = np.rollaxis(cube, dimension, 0)
        print str(dimension) + " -> " + str(cube.shape)

        # nn_result = np.zeros(cube.shape, dtype=np.double)
        #
        # for i in xrange(0, cubeshape[dimension]):
        #
        #     if i == 0:
        #         # Reflect the image to account for the border problem
        #         indata = np.zeros((3, cube.shape[1], cube.shape[2]), dtype=cube.dtype)
        #         indata[1:3, :, :] = cube[i:i + 2, :, :]
        #         indata[0, :, :] = cube[i + 2, :, :]
        #     elif i == cubeshape[dimension]-1:
        #         # Reflect the image to account for the border problem
        #         indata = np.zeros((3, cube.shape[1], cube.shape[2]), dtype=cube.dtype)
        #         indata[0:2, :, :] = cube[i-1:i+1, :, :]
        #         indata[2, :, :] = cube[i - 1, :, :]
        #     else:
        #         indata = cube[i - 1:i + 2, :, :]
        #
        #     print indata.shape
        #
        #     nn_result_t = self.call_nn(indata)
        #     nn_result[i, :, :] = nn_result_t[1, :, :]
        # print i

        # # Roll the result back to obtain the dimensional orientation of the input data
        # nn_result = np.rollaxis(nn_result, 0, dimension+1)
        # print nn_result.shape
        #
        # return nn_result

    def nn_on_cube(self, input, resultfile, cubename='cube'):

        input = np.zeros((10, 20, 30))

        for dim in xrange(0, 3):
            self.nn_on_cube_dimension(input, dim)
            # print str(dim) + " -> " + str(input.shape)

    def call_nn(self, input):
        # TODO: Put NN function here!
        return input

    def create_cube(self, position=(0,0,0), size=512, step=32):

        x = range(0, (size/step)**3, 1)
        x[:] = [i / (size/step)**2 * step + position[0] for i in x]
        y = range(0, (size/step)**2, 1)
        y[:] = [i / (size/step) * step + position[1] for i in y] * (size/step)
        z = range(0, size, step) * (size/step) ** 2
        z[:] = [i + position[2] for i in z]

        # print x
        # print y
        # print z

        cube = (x, y, z)

        cube = map(list, zip(*cube))

        cube = [tuple(el) for el in cube]

        return cube

    def move_cube(self, position, cube, cube_init):
        # print square
        # print position

        for i in xrange(0, len(cube)):
            cube[i] = (cube_init[i][0] + position[0], cube_init[i][1] + position[1], cube_init[i][2] + position[2])

        return cube

        # print square

    def create_square(self, position=[0, 0, 0], size=512, step=32):

        x = range(0, (size/step)**3, 1)
        x[:] = [i / (size/step)**2 * step + position[0] for i in x]
        y = range(0, (size/step)**2, 1)
        y[:] = [i / (size/step) * step + position[1] for i in y] * (size/step)
        z = range(0, size, step) * (size/step) ** 2
        z[:] = [i + position[2] for i in z]

        # print x
        # print y
        # print z

        square = [x, y, z]

        square = map(list, zip(*square))

        return square

    def move_square(self, position, square, square_init):
        # print square
        # print position

        for i in xrange(0, len(square)):
            square[i][0] = square_init[i][0] + position[0]
            square[i][1] = square_init[i][1] + position[1]
            square[i][2] = square_init[i][2] + position[2]

        return square

        # print square

    def rois_dict(self, rois_in, size=1024, overlap=512):

        rois_out = {}
        a= 10
        for i in rois_in:
            x = i[0] / (size-overlap) * (size-overlap)
            y = i[1] / (size-overlap) * (size-overlap)
            z = i[2] / (size-overlap) * (size-overlap)
            if (x, y, z) in rois_out.keys():
                rois_out[(x, y, z)] += [i]
            else:
                rois_out[(x, y, z)] = [i]
            # if overlap > 0:
            #     x2 = (i[0]-overlap) / (size-overlap) * (size-overlap)
            #     y2 = (i[1]-overlap) / (size-overlap) * (size-overlap)
            #     z2 = (i[2]-overlap) / (size-overlap) * (size-overlap)
            #     if x != x2 or y != y2 or z != z2:
            #         if (x2, y2, z2) in rois_out.keys():
            #             rois_out[(x2, y2, z2)] += [i]
            #         else:
            #             rois_out[(x2, y2, z2)] = [i]

        return rois_out

    def open_rois(self, input="/media/julian/Daten/mobi/h1.hci/data/fib25/rois.pkl"):
        f = open(input)
        data = pickle.load(f)
        return data

    def detect_rois_dict2(self, input="/media/julian/Daten/mobi/h1.hci/data/fib25/rois.pkl",
                    size=512, step=32, dict_size=1025, dict_overlap=512):

        datasize = [2623, 6527, 3168]

        data = self.open_rois(input)

        areas = self.rois_dict(data, size=dict_size, overlap=dict_overlap)
        print areas.keys()

        # data = data[0:100000]

        print data[0]
        # size = 512
        # step = 32

        square_init = self.create_square(position=[0, 0, 0], size=size, step=step)
        square = copy.deepcopy(square_init)

        complete = 0

        for areakey in areas.keys():
            # TODO: This doesn't work...
            c_area = areas[areakey]
            w_areakeys = [areakey,
                          (areakey[0]+dict_size, areakey[1], areakey[2]),
                          (areakey[0], areakey[1]+dict_size, areakey[2]),
                          (areakey[0], areakey[1], areakey[2]+dict_size),
                          (areakey[0]+dict_size, areakey[1]+dict_size, areakey[2]),
                          (areakey[0], areakey[1]+dict_size, areakey[2]+dict_size),
                          (areakey[0]+dict_size, areakey[1], areakey[2]+dict_size),
                          (areakey[0]+dict_size, areakey[1]+dict_size, areakey[2]+dict_size)]
            print "areakey:"

            # print w_areakeys
            w_areakeys = set(w_areakeys).intersection(set(areas.keys()))
            w_area=[]
            for key in w_areakeys:
                w_area += areas[key]
            print w_area[0:10]
            # print areakey
            # print w_areakeys
            while len(c_area) > 0:

                if (complete % 100) == 0:
                    print str(complete) + '/' + str(len(data))
                complete += 1

                d = data[0]

                square = self.move_square(position=d, square=square, square_init=square_init)

                if (square[0][0] < datasize[0]-size) \
                        and (square[0][1] < datasize[1]-size)\
                        and (square[0][2] < datasize[2]-size):

                    # TODO: This doesn't work...
                    count = 0
                    val = square[0]
                    while val in w_area and count < (size / step) ** 3:
                        val = square[count]
                        count += 1
                    print count
                    if count == (size / step) ** 3:

                        for val in square:
                            data.remove(val)

                        print 'Found cube!'
                        print square
                        print len(data)

                    else:
                        data.pop(0)

                else:
                    data.pop(0)

    def detect_rois_dict(self, input="/media/julian/Daten/mobi/h1.hci/data/fib25/rois.pkl",
                    size=512, step=32, dict_size=1025, dict_overlap=512):

        datasize = [2623, 6527, 3168]

        data = self.open_rois(input)

        areas = self.rois_dict(data, size=dict_size, overlap=dict_overlap)
        print areas.keys()

        # data = data[0:100000]

        print data[0]
        # size = 512
        # step = 32

        square_init = self.create_square(position=[0, 0, 0], size=size, step=step)
        square = copy.deepcopy(square_init)

        complete = 0

        while len(data) > 0:

            if (complete % 100) == 0:
                print str(complete) + '/' + str(len(data))
            complete += 1

            d = data[0]

            square = self.move_square(position=d, square=square, square_init=square_init)

            if (square[0][0] < datasize[0]-size) \
                    and (square[0][1] < datasize[1]-size)\
                    and (square[0][2] < datasize[2]-size):


                # print d
                # print d[0] / (dict_size-dict_overlap) * (dict_size-dict_overlap)
                # print d[1] / (dict_size-dict_overlap) * (dict_size-dict_overlap)
                # print d[2] / (dict_size-dict_overlap) * (dict_size-dict_overlap)
                area = areas[(d[0] / (dict_size-dict_overlap) * (dict_size-dict_overlap),
                              d[1] / (dict_size-dict_overlap) * (dict_size-dict_overlap),
                              d[2] / (dict_size-dict_overlap) * (dict_size-dict_overlap))]


                # print d
                # print area[0]
                # print len(area)
                if (d[0] / dict_size * dict_size + dict_size, d[1] / dict_size * dict_size, d[2] / dict_size * dict_size) in areas.keys():
                    area += areas[(d[0] / dict_size * dict_size + dict_size, d[1] / dict_size * dict_size, d[2] / dict_size * dict_size)]
                if (d[0] / dict_size * dict_size, d[1] / dict_size * dict_size + dict_size, d[2] / dict_size * dict_size) in areas.keys():
                    area += areas[(d[0] / dict_size * dict_size, d[1] / dict_size * dict_size + dict_size, d[2] / dict_size * dict_size)]
                if (d[0] / dict_size * dict_size, d[1] / dict_size * dict_size, d[2] / dict_size * dict_size + dict_size) in areas.keys():
                    area += areas[(d[0] / dict_size * dict_size, d[1] / dict_size * dict_size, d[2] / dict_size * dict_size + dict_size)]
                if (d[0] / dict_size * dict_size + dict_size, d[1] / dict_size * dict_size + dict_size, d[2] / dict_size * dict_size) in areas.keys():
                    area += areas[(d[0] / dict_size * dict_size + dict_size, d[1] / dict_size * dict_size + dict_size, d[2] / dict_size * dict_size)]
                if (d[0] / dict_size * dict_size, d[1] / dict_size * dict_size + dict_size, d[2] / dict_size * dict_size + dict_size) in areas.keys():
                    area += areas[(d[0] / dict_size * dict_size, d[1] / dict_size * dict_size + dict_size, d[2] / dict_size * dict_size + dict_size)]
                if (d[0] / dict_size * dict_size + dict_size, d[1] / dict_size * dict_size, d[2] / dict_size * dict_size + dict_size) in areas.keys():
                    area += areas[(d[0] / dict_size * dict_size + dict_size, d[1] / dict_size * dict_size, d[2] / dict_size * dict_size + dict_size)]
                if (d[0] / dict_size * dict_size + dict_size, d[1] / dict_size * dict_size + dict_size, d[2] / dict_size * dict_size + dict_size) in areas.keys():
                    area += areas[(d[0] / dict_size * dict_size + dict_size, d[1] / dict_size * dict_size + dict_size, d[2] / dict_size * dict_size + dict_size)]

                count = 0
                val = square[0]
                while val in area and count < (size / step) ** 3:
                    val = square[count]
                    count += 1

                if count == (size / step) ** 3:

                    for val in square:
                        data.remove(val)

                    print 'Found cube!'
                    print square
                    print len(data)

                else:
                    data.pop(0)

            else:
                data.pop(0)

    def store_rois(self, file, rois):
        pickle.dump(rois, open(file, "wb"), protocol=pkl.HIGHEST_PROTOCOL)

    def extract_rois(self, inputfile="/media/julian/Daten/mobi/h1.hci/data/fib25/rois.pkl",
                     outputfile="/media/julian/Daten/mobi/h1.hci/data/fib25/rois512.pkl",
                     size=512, step=32):
        """Detects ROIs of given size and saves them as pickle"""

        rois = self.detect_rois(input=inputfile, size=size, step=step)
        self.store_rois(file=outputfile, rois=rois)

    def detect_rois2(self, input="/media/julian/Daten/mobi/h1.hci/data/fib25/rois.pkl",
                    size=512, step=32):
        """Uses tuples for coordinate representation"""

        datasize = [2623, 6527, 3168]

        data = self.open_rois(input)
        data = [tuple(el) for el in data]

        # data = data[0:100000]

        # print data[0:10]

        cube_init = self.create_cube(position=[0, 0, 0], size=size, step=step)
        cube = copy.deepcopy(cube_init)

        complete = 0

        rois_out = []

        while len(data) > 0:

            if (complete % 100) == 0:
                print str(complete) + '/' + str(len(data))
            complete += 1

            d = data[0]

            cube = self.move_cube(position=d, cube=cube, cube_init=cube_init)

            if (cube[0][0] < datasize[0]-size) \
                    and (cube[0][1] < datasize[1]-size)\
                    and (cube[0][2] < datasize[2]-size):

                count = 0
                val = cube[0]
                while val in data and count < (size / step) ** 3:
                    val = cube[count]
                    count += 1

                if count == (size / step) ** 3:

                    [data.remove(val) for val in cube]

                    print 'Found cube!'
                    print cube
                    print len(data)

                    # Store only the first coordinate of the cube, the size information is known
                    rois_out += cube[0]

                else:
                    data.pop(0)
            else:
                data.pop(0)

        return rois_out

    def get_upper_bounds(self, rois, step):

        bounds = [0,0,0]

        bounds[0] = max([x[0] for x in rois]) + 32
        bounds[1] = max([x[1] for x in rois]) + 32
        bounds[2] = max([x[2] for x in rois]) + 32

        return bounds

    def get_lower_bounds(self, rois):

        bounds = [0,0,0]

        bounds[0] = min([x[0] for x in rois])
        bounds[1] = min([x[1] for x in rois])
        bounds[2] = min([x[2] for x in rois])

        return bounds

    def alternating_coord(self, x, min, len):

        if x & 1:
            # Uneven number
            return (len - (x-min/2*2)) + min/2*2
        else:
            # Even number
            return x

    def detect_rois(self, input="/media/julian/Daten/mobi/h1.hci/data/fib25/rois.pkl",
                    size=512, step=32):
        """Uses lists for coordinate representation"""

        # datasize = [2623, 6527, 3168]

        data = sorted(self.open_rois(input))

        bounds_u = self.get_upper_bounds(data, step)
        bounds_l = self.get_lower_bounds(data)

        print "Detected upper bounds: " + str(bounds_u)
        print "Detected lower bounds: " + str(bounds_l)

        cube_init = self.create_square(position=[0, 0, 0], size=size, step=step)
        cube = copy.deepcopy(cube_init)

        complete = 0

        rois_out = []

        alternating_coords = [self.alternating_coord(x, 0, len(cube)) for x in range(0, len(cube))]

        # # Temporarily abort...
        # return rois_out

        while len(data) > 0:

            if (complete % 100) == 0:
                print str(complete) + '/' + str(len(data))
            complete += 1

            d = data[0]

            if (d[0] < bounds_u[0]-size) \
                    and (d[1] < bounds_u[1]-size)\
                    and (d[2] < bounds_u[2]-size):

                cube = self.move_square(position=d, square=cube, square_init=cube_init)
                count = 0

                cubelen = len(cube)
                val = cube[0]
                while val in data and count < (size / step) ** 3:
                    val = cube[alternating_coords[count]]
                    count += 1

                if count == (size / step) ** 3:

                    # for val in cube:
                    #     data.remove(val)
                    [data.remove(val) for val in cube]

                    print 'Found cube!'
                    print cube
                    print str(complete) + '/' + str(len(data))

                    # Store only the first coordinate of the cube, the size information is known
                    rois_out += cube[0]

                else:
                    data.pop(0)
            else:
                data.pop(0)

        return rois_out


if __name__ == "__main__":

    path = "/media/julian/Daten/mobi/h1.hci/data/fib25/raw_fib25.h5"
    datapath = "data"
    resultfile = "/media/julian/Daten/mobi/h1.hci/data/fib25/results/result_fib25"
    #
    nnupsc = nn_upscale(path=path, datapath=datapath, resultfile=resultfile)

    # nnupsc.extract_rois("/media/julian/Daten/mobi/h1.hci/data/fib25/rois.pkl",
    #                     "/media/julian/Daten/mobi/h1.hci/data/fib25/rois512.pkl",
    #                     size=512, step=32)


    nnupsc.train_nn(roispath="/media/julian/Daten/mobi/h1.hci/data/fib25/roissl512.pkl",
                    popcubes=10)

    # nnupsc.detect_rois_dict2(dict_size=4068, dict_overlap=0)
    # nnupsc.detect_rois_dict(dict_size=512, dict_overlap=0)

    # nnupsc.detect_rois()
