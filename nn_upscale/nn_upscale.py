
from netdatautils import fromh5, slidingwindowslices
# import scipy
import time
import copy
import pickle
import math

__author__ = 'jhennies'


class nn_upscale:

    _path = ""
    _datapath = ""
    _currentslice=None
    _datasetsize=None
    _nhoodsize=None
    _stride=None

    def __init__(self):
        pass

    def __init__(self, path=None, datapath=None,
                 datasetsize=[256, 256, 256],
                 nhoodsize=[32, 32, 32],
                 stride=16):
        self._path = path
        self._datapath = datapath
        self._datasetsize = datasetsize
        self._nhoodsize = nhoodsize
        self._stride = stride

    def run_nn_on_dataset(self):

        # Determine slices which will be used
        slices = slidingwindowslices(self._datasetsize, self._nhoodsize, stride=self._stride, shuffle=False)

        # Iterate over the slices
        count = 0
        for currentslice in slices:
            count += 1
            print "slice:"
            print (currentslice[0], currentslice[1], currentslice[2])

            # Get the necessary part of the image
            data = self.load_data(currentslice)
            print data.shape

            # Compute the neuronal network
            output = self.call_nn(input=data)

        # And it's done
        print "The network was performed on " + str(count) + "images. "

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

    def rois_dict(self, rois_in, size=1024):

        rois_out = {}

        for i in rois_in:
            x = i[0] / size * size
            y = i[1] / size * size
            z = i[2] / size * size
            if (x, y, z) in rois_out.keys():
                rois_out[(x, y, z)] += [i]
            else:
                rois_out[(x, y, z)] = [i]

        return rois_out

    def open_rois(self, input="/media/julian/Daten/mobi/h1.hci/data/fib25/rois.pkl"):
        f = open(input)
        data = pickle.load(f)
        return data

    def detect_rois(self, input="/media/julian/Daten/mobi/h1.hci/data/fib25/rois.pkl", size=512, step=32):

        datasize = [2623, 6527, 3168]

        data = self.open_rois(input)

        # data = data[0:100000]

        print data[0]
        size = 512
        step = 32

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
                count = 0
                val = square[0]
                while val in data and count < (size / step) ** 3:
                    val = square[count]
                    count += 1

                if count == (size / step) ** 3:

                    for val in square:
                        data.remove(val)
                        # process_data.remove(val)

                    print 'Found cube!'
                    print square
                    print len(data)

                else:
                    data.pop(0)
            else:
                data.pop(0)


    def load_data(self, currentslice):
        return fromh5(self._path, self._datapath, currentslice, True, None)

    def call_nn(self, input=None):

        # TODO: extract the slices

        # TODO: function to call the NN

        return input


if __name__ == "__main__":

    path = "/media/julian/Daten/mobi/h1.hci/data/testdata/zeros.h5"
    datapath = "zeros"
    #
    nnupsc = nn_upscale(path=path, datapath=datapath)
    # nnupsc.run_nn_on_dataset()

    nnupsc.detect_rois()

    # sq = nnupsc.create_square(position=[0,0,0], size=10, step=5)
    # print sq
    #
    # sq20 = nnupsc.create_square(position=[20, 30, 40], size=10, step=5)
    # print sq20
    #
    # sq20m = nnupsc.move_square(position=[20, 30, 40], square=sq)
    # print sq20m

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


