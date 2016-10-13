
from hdf5_processing import Hdf5Processing
import numpy as np
import processing_lib as lib

__author__ = 'jhennies'


class Hdf5ImageProcessing(Hdf5Processing):

    def __init__(self, *args, **kwargs):
        super(Hdf5ImageProcessing, self).__init__(*args, **kwargs)

    # def __setitem__(self, key, val):
    #     if type(key) is tuple or type(key) is list:
    #         if len(key) > 1:
    #             key = list(key)
    #             fkey = key.pop(0)
    #             self[fkey][key] = val
    #         else:
    #             super(Hdf5ImageProcessing, self).__setitem__(key[0], val)
    #     else:
    #         super(Hdf5ImageProcessing, self).__setitem__(key, val)

    def anytask(self, task, *args, **kwargs):
        """
        :param task: The function which will be executed (has to take an image as first argument or first two arguments)
        :param ids=None: The dictionary entries which will be used for calculation
        :param args: Remaining arguments of task
        :param kwargs: Remaining keyword arguments of task

        :return:
        """

        # Defaults
        keys = None
        keys2 = None
        targetkeys = None

        # Get keyword arguments
        if 'keys' in kwargs.keys():
            keys = kwargs.pop('keys')
            if type(keys) is not tuple and type(keys) is not list:
                keys = (keys,)
        if keys is None:
            keys = self.keys()

        for k in keys:

            # print 'Working on key {}'.format(k)
            if type(k) is tuple:
                k = list(k)
            if type(k) is not list:
                k = [k,]

            print 'k = {}'.format(k)
            print 'keys = {}'.format(self.keys())

            # currentk = k.pop(0)

            if type(self[k]) is not type(self):
                task(self[k], *args, **kwargs)
            else:
                self[k].anytask(task, *args, **kwargs)


class Hdf5ImageProcessingLib(Hdf5ImageProcessing):

    def __init__(self, *args, **kwargs):
        super(Hdf5ImageProcessingLib, self).__init__(*args, **kwargs)


if __name__ == '__main__':

    # hp = Hdf5Processing()
    hipl = Hdf5ImageProcessingLib()

    hipl['a', 'b', 'c1'] = np.zeros((10, 10, 10))
    hipl['a', 'b', 'c2'] = np.ones((10, 10, 10))

    hipl['a', 'b2', 'd'] = np.ones((10, 10, 10))*2
    hipl['a', 'b2', 'e'] = np.ones((10, 10, 10))*3

    hipl['f', 'g', 'h'] = np.ones((10, 10, 10))*5

    print hipl.datastructure2string()

    print type(hipl)
    print type(hipl['a'])
    print type(hipl['a','b'])
    print type(hipl['a','b','c1'])

    hipl.anytask(lib.add, 10, keys=(('a', 'b'), 'f'))
