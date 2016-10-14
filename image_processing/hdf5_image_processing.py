
from hdf5_processing import Hdf5Processing
import numpy as np
import processing_lib as lib

__author__ = 'jhennies'


class Hdf5ImageProcessing(Hdf5Processing):

    def __init__(self, *args, **kwargs):
        super(Hdf5ImageProcessing, self).__init__(*args, **kwargs)

    # def anytask(self, task, *args, **kwargs):
    #     """
    #     :param task: The function which will be executed (has to take an image as first argument or first two arguments)
    #     :param ids=None: The dictionary entries which will be used for calculation
    #     :param args: Remaining arguments of task
    #     :param kwargs: Remaining keyword arguments of task
    #
    #     :return:
    #     """
    #
    #     # Defaults
    #     keys = None
    #     keys2 = None
    #     targetkeys = None
    #
    #     # Get keyword arguments
    #     if 'keys' in kwargs.keys():
    #         keys = kwargs.pop('keys')
    #         if type(keys) is not tuple and type(keys) is not list:
    #             keys = (keys,)
    #     if keys is None:
    #         keys = self.keys()
    #
    #     for k in keys:
    #
    #         # print 'Working on key {}'.format(k)
    #         if type(k) is tuple:
    #             k = list(k)
    #         if type(k) is not list:
    #             k = [k,]
    #
    #         print 'k = {}'.format(k)
    #         print 'keys = {}'.format(self.keys())
    #
    #         if type(self[k]) is not type(self):
    #             self[k] = task(self[k], *args, **kwargs)
    #         else:
    #             self[k] = self[k].anytask(task, *args, **kwargs)
    #
    #     return self

    def anytask(self, task, *args, **kwargs):
        """
        :param task: The function which will be executed (has to take an image as first argument or first two arguments)
        :param keys=None: The dictionary entries which will be used for calculation
        :param tkeys=None: The target keys, i.e. the dictionary entries which will be used for storage of the result
            By default the input data will be overwritten
            Set tkeys to empty list or tuple to enable return only
        :param args: Remaining arguments of task
        :param kwargs: Remaining keyword arguments of task

        :return:
        """

        # Defaults
        keys = None
        keys2 = None
        tkeys = None

        # Get keys from input
        if 'keys' in kwargs.keys():
            keys = kwargs.pop('keys')
            if type(keys) is not tuple and type(keys) is not list:
                keys = (keys,)
        if keys is None:
            keys = self.keys()

        # Get targetkeys from input
        if 'tkeys' in kwargs.keys():
            tkeys = kwargs.pop('tkeys')
            # Make sure tkeys is either tuple or list
            if type(tkeys) is not tuple and type(tkeys) is not list:
                tkeys = (tkeys,)
            # Check for appropriate length: Either equal to len(keys) ...
            if len(tkeys) != len(keys):
                # ... or one
                if len(tkeys) == 1:
                    # Create a tuple list which is the same length as keys ...
                    tkeys *= len(keys)
                    # ... and zip it to the keys such that a new result entry containing all the keys from the source
                    # entries is created
                    tkeys = zip(tkeys, keys)
                else:
                    raise RuntimeError('Hdf5ImageProcessing.anytask: The length of keys and tkeys has to be identical!')
        if tkeys is None:
            tkeys = keys

        # Zip-it to make simultaneous iteration over both keys and tkeys easier
        akeys = zip(keys, tkeys)

        for k in akeys:

            if type(self[k[0]]) is not type(self):
                self[k[1]] = task(self[k[0]], *args, **kwargs)
            else:
                self[k[1]] = self[k[0]].anytask(task, *args, **kwargs)

        return self

class Hdf5ImageProcessingLib(Hdf5ImageProcessing):

    def __init__(self, *args, **kwargs):
        super(Hdf5ImageProcessingLib, self).__init__(*args, **kwargs)


if __name__ == '__main__':

    # hp = Hdf5Processing()
    hipl = Hdf5ImageProcessingLib()

    hipl['a', 'b', 'c1'] = np.zeros((10, 10))
    hipl['a', 'b', 'c2'] = np.ones((10, 10))

    hipl['a', 'b2', 'd'] = np.ones((10, 10))*2
    hipl['a', 'b2', 'e'] = np.ones((10, 10))*3

    hipl['f', 'g', 'h'] = np.ones((10, 10))*5

    print hipl.datastructure2string()

    print type(hipl)
    print type(hipl['a'])
    print type(hipl['a','b'])
    print type(hipl['a','b','c1'])

    hipl.anytask(lib.add, 10, keys=(('a', 'b'), 'f'))
    hipl.anytask(lib.mult, 0.5)

    print hipl['a', 'b', 'c1'][0, 0]
    print hipl['a', 'b', 'c2'][0, 0]
    print hipl['f','g','h'][0, 0]
    print hipl['a', 'b2', 'd'][0, 0]
    print hipl['a', 'b2', 'e'][0, 0]

    hipl.anytask(lib.add, 20, keys=(('a', 'b2'), ('f', 'g')), tkeys=(('a', 'b3'), 'f2'))

    print hipl.datastructure2string()
    print '---'

    hipl.anytask(lib.add, 7, tkeys=('result',))

    print hipl.datastructure2string()