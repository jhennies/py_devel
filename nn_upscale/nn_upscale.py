
from netdatautils import fromh5, toh5, slidingwindowslices
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

    def run_nn_on_dataset(self, roispath, cubesize=512):

        # # Determine slices which will be used
        # slices = slidingwindowslices(self._datasetsize, self._nhoodsize, stride=self._stride, shuffle=False)
        #
        # # Iterate over the slices
        # count = 0
        # for currentslice in slices:
        #     count += 1
        #     print "slice:"
        #     print (currentslice[0], currentslice[1], currentslice[2])
        #
        #     # Get the necessary part of the image
        #     data = self.load_data(currentslice)
        #     print data.shape
        #
        #     # Compute the neuronal network
        #     output = self.call_nn(input=data)
        #
        # # And it's done
        # print "The network was performed on " + str(count) + "images. "


        # -------------------------------------
        # New version:

        # Load ROIs
        rois = self.open_rois(input=roispath)
        rois = zip(*[iter(rois)]*3)
        print len(rois)

        rois = rois[0:1]
        # Iterate over ROIs
        for roi in rois:
            # print roi
            # Get the slice information
            sl = (slice(roi[0], roi[0]+cubesize),
                  slice(roi[1], roi[1]+cubesize),
                  slice(roi[2], roi[2]+cubesize))

            print "slice:"
            print sl

            # Load the respective part of the image
            data = self.load_data(sl)
            # print data[0:10, 0:10, 0:10]

            # Compute the neuronal network
            self.nn_on_cube(input=data, resultfile=self._resultfile, cubename=str(roi[0]) + "_" + str(roi[1]) + "_" + str(roi[2]))

        # It's done

    def load_data(self, currentslice):
        return fromh5(self._path, datapath=self._datapath,
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


    nnupsc.run_nn_on_dataset("/media/julian/Daten/mobi/h1.hci/data/fib25/rois512.pkl", cubesize=512)

    # nnupsc.detect_rois_dict2(dict_size=4068, dict_overlap=0)
    # nnupsc.detect_rois_dict(dict_size=512, dict_overlap=0)

    # nnupsc.detect_rois()


    # # ------------------------
    # # Testing tuples...
    # sq = nnupsc.create_square()
    # sq_init = copy.deepcopy(sq)
    # start = time.clock()
    # sq = nnupsc.move_square([32, 32, 32], sq, sq_init)
    # end = time.clock()
    # print "Elapsed time: " + str(end-start)
    # print sq
    #
    # cube = nnupsc.create_cube()
    # cube_init = copy.deepcopy(cube)
    # start = time.clock()
    # cube = nnupsc.move_cube((32, 32, 32), cube, cube_init)
    # end = time.clock()
    # print "Elapsed time: " + str(end-start)
    # print cube
    # # ------------------------

    # sq = nnupsc.create_square(position=[0,0,0], size=10, step=5)
    # print sq
    #
    # sq20 = nnupsc.create_square(position=[20, 30, 40], size=10, step=5)
    # print sq20
    #
    # sq20m = nnupsc.move_square(position=[20, 30, 40], square=sq, square_init=sq)
    # print sq20m
    #
    # tsq = tuple(sq)
    # print tsq
    #
    # itsq = [tuple(el) for el in sq]
    # print itsq
    #
    # set(itsq)
    # print set(itsq)

    # rois = nnupsc.rois_dict(nnupsc.open_rois())
    # print rois.keys()

    # start = time.clock()
    #
    # sq = nnupsc.create_square()
    #
    # end = time.clock()
    # print 'Time elapsed:'
    # print end - start
    #
    # start = time.clock()
    #
    # nnupsc.move_square([32, 64, 128], sq)
    #
    # end = time.clock()
    # print 'Time elapsed:'
    # print end - start

    # f = open("/media/julian/Daten/mobi/h1.hci/data/fib25/rois.pkl")
    # data = pickle.load(f)
    #
    # print data[0]
    # size = 512
    # step = 32
    #
    # complete = 0
    #
    # lendata = len(data)
    #
    # # process_data = copy.deepcopy(data)
    #
    # # for d in data:
    # i = 0
    # while len(data) > 0:
    #
    #     if (complete % 100) == 0:
    #         print str(complete) + '/' + str(len(data))
    #     complete += 1
    #
    #     d = data[0]
    #
    #     square = nnupsc.create_square(position=d, size=size, step=step)
    #     # print square
    #     # matches = set(data) & set(square)
    #     # matches = set(data).intersection(square)
    #     # print matches
    #
    #
    #     # # This is slow:
    #     # start = time.clock()
    #     #
    #     # count = 0
    #     # for val in data:
    #     #     if val in square:
    #     #         count += 1
    #     #
    #     # print count
    #     #
    #     # end = time.clock()
    #     # print 'Time elapsed:'
    #     # print end - start
    #
    #     # start = time.clock()
    #
    #     # Finds matches in the square and the data
    #     # count = 0
    #     # for val in square:
    #     #     if val in data:
    #     #         count += 1
    #     #
    #     # print count
    #
    #     count = 0
    #     val = square[0]
    #     while val in data and count < (size/step)**3:
    #         val = square[count]
    #         count += 1
    #
    #     # print count
    #
    #     # end = time.clock()
    #     # print 'Time elapsed:'
    #     # print end - start
    #
    #     # print len(data)
    #
    #     if count == (size/step)**3:
    #
    #         for val in square:
    #             data.remove(val)
    #             # process_data.remove(val)
    #
    #         print 'Found cube!'
    #         print square
    #         print len(data)
    #
    #     else:
    #         # process_data = process_data[1:]
    #         data.pop(0)
    #
    #     # print len(data)
    #
    #     # print len(data)
    #     # print "------------------------"
    #
    #
    #
    #     # for a in xrange(1, len(data)):
    # #     print a


