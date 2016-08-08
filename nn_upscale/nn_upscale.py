
from netdatautils import fromh5, slidingwindowslices
# import scipy

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

    def load_data(self, currentslice):
        return fromh5(self._path, self._datapath, currentslice, True, None)

    def call_nn(self, input=None):

        # TODO: extract the slices

        # TODO: function to call the NN

        return input


if __name__ == "__main__":

    path = "/media/julian/Daten/mobi/h1.hci/data/testdata/zeros.h5"
    datapath = "zeros"

    nnupsc = nn_upscale(path=path, datapath=datapath)
    nnupsc.run_nn_on_dataset()


