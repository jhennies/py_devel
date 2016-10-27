
from hdf5_processing import Hdf5Processing
import numpy as np
import processing_lib as lib
import numpy as np
from copy import deepcopy
# import numpy.lib.index_tricks

__author__ = 'jhennies'


###############################################################################################

class Hdf5ImageProcessing(Hdf5Processing):

    def __init__(self, *args, **kwargs):
        super(Hdf5ImageProcessing, self).__init__(*args, **kwargs)

    def reciprocal_key_gen(self, names1, names2):

        lenk = len(names1)
        names1 *= len(names2)
        names2 = tuple(sorted(names2 * lenk))

        return self.resultkey_name_gen(names1, names2)

    @staticmethod
    def resultkey_name_gen(names1, names2):

        names = zip(names1, names2)
        tkeys = ()
        for k in names:
            kstr = ''
            for l in k:
                if type(l) is tuple or type(l) is list:
                    for m in l:
                        if kstr != '':
                            kstr += '_' + str(m)
                        else:
                            kstr += str(m)
                else:
                    if kstr != '':
                        kstr += '_' + str(l)
                    else:
                        kstr += str(l)
            tkeys += (kstr,)
        return tkeys

    def anytask(self, task, *args, **kwargs):
        """
        :param task: The function which will be executed (has to take an image as first argument or first two arguments)
        :param keys=None: The dictionary entries which will be used for calculation
        :param keys2=None: Additional lot of dictionary entries used for calculation in functions taking two data
            arguments
        :param indict=None: Real dictionary-equivalent of keys2; can be any other dictionary of subset of self
            indict is set to self if omitted and keys2 argument is given
        :param tkeys=None: The target keys, i.e. the dictionary entries which will be used for storage of the result
            By default the input data will be overwritten
            Set tkeys to empty list or tuple to enable return only
        :param reciprocal=False: When False, each entry of keys is computed with the respective entry in keys2 (Make sure
            keys and keys2 have the same structure!)
            When True, each element of keys is computed with each element of keys2
            Use with care!
        :param args: Remaining arguments of task
        :param kwargs: Remaining keyword arguments of task

        :return:
        """

        # Defaults
        keys = None
        keys2 = None
        tkeys = None
        indict = None
        reciprocal = False
        reverse_order = False
        returnonly = False

        # Get flags
        if 'reciprocal' in kwargs.keys():
            reciprocal = kwargs.pop('reciprocal')
        if 'reverse_order' in kwargs.keys():
            reverse_order = kwargs.pop('reverse_order')
        if 'return_only' in kwargs.keys():
            returnonly = kwargs.pop('return_only')

        # Get keys from input
        if 'keys' in kwargs.keys():
            keys = kwargs.pop('keys')
            if type(keys) is not tuple and type(keys) is not list:
                keys = (keys,)
        if keys is None:
            keys = self.keys()

        # Get keys2 from input
        if 'keys2' in kwargs.keys():
            keys2 = kwargs.pop('keys2')
            if type(keys2) is not tuple and type(keys2) is not list:
                keys2 = (keys2,)

        # Get indict from input
        if 'indict' in kwargs.keys():
            indict = kwargs.pop('indict')
            keys2 = indict.keys()

        if keys2 is not None:
            # When supplied, check for appropriate length or make reciprocal key lists
            if reciprocal:
                lenk = len(keys)
                keys *= len(keys2)
                keys2 = tuple(sorted(keys2 * lenk))
                # print keys
                # print keys2
            else:
                if len(keys) != len(keys2):
                    if len(keys2) == 1:
                        keys2 *= len(keys)
                    else:
                        raise RuntimeError('Hdf5ImageProcessing.anytask: The length of keys and keys2 has to be identical in this setting!')

        # Get targetkeys from input
        if 'tkeys' in kwargs.keys():
            tkeys = kwargs.pop('tkeys')
            # Make sure tkeys is either tuple or list
            if type(tkeys) is not tuple and type(tkeys) is not list:
                tkeys = (tkeys,)
        if tkeys is None:

            if reciprocal:

                tkeys = self.resultkey_name_gen(keys, keys2)

            else:
                tkeys = keys

        if len(tkeys) != len(keys):

            if len(tkeys) == 1:

                if reciprocal:

                    ttkeys = self.resultkey_name_gen(keys, keys2)
                    tkeys = zip(tkeys*len(ttkeys), ttkeys)

                else:

                    tkeys = tuple([tkeys + (x,) for x in keys])

            else:
                raise RuntimeError('Hdf5ImageProcessing.anytask: The length of keys and tkeys has to be identical!')

        if returnonly:
            rtrn = type(self)()
        else:
            rtrn = self

        if keys2 is None:

            # This is performed if neither keys2 nor indict is given
            akeys = zip(keys, tkeys)
            for k in akeys:

                if type(self[k[0]]) is not type(self):
                    rtrn[k[1]] = task(self[k[0]], *args, **kwargs)
                else:
                    rtrn[k[1]] = self[k[0]].anytask(task, *args, **kwargs)

        else:

            if indict is None:
                indict = self
            akeys = zip(keys, keys2, tkeys)
            for k in akeys:

                if type(self[k[0]]) is not type(self):
                    if reverse_order:
                        rtrn[k[2]] = task(indict[k[1]], self[k[0]], *args, **kwargs)
                    else:
                        rtrn[k[2]] = task(self[k[0]], indict[k[1]], *args, **kwargs)
                else:
                    rtrn[k[2]] = self[k[0]].anytask(task, *args, indict=indict[k[1]], **kwargs)

        return rtrn

    def astype(self, dtype, **kwargs):
        self.anytask(lib.astype, dtype, **kwargs)

    def deepcopy_entry(self, skey, tkey):
        self[tkey] = deepcopy(self[skey])

    def rename_entry(self, old, new):
        # self[new] = self[old]
        # del self[old]
        self[new] = self.pop(old)

###############################################################################################

class Hdf5ImageProcessingLib(Hdf5ImageProcessing):

    def __init__(self, *args, **kwargs):
        super(Hdf5ImageProcessingLib, self).__init__(*args, **kwargs)

    # _________________________________________________________________________________________
    # Image processing

    def positions2value(self, coordinates, value, **kwargs):
        return self.anytask(lib.positions2value, coordinates, value, **kwargs)

    def unique(self, **kwargs):
        return self.anytask(lib.unique, **kwargs)

    def getlabel(self, label, **kwargs):
        return self.anytask(lib.getlabel, label, **kwargs)

    def filter_values(self, value, **kwargs):
        return self.anytask(lib.filter_values, value, **kwargs)

    def pixels_at_boundary(self, axes=[1, 1, 1], **kwargs):
        return self.anytask(lib.pixels_at_boundary, axes=axes, **kwargs)

    def distance_transform(self, pixel_pitch=(), background=True, **kwargs):
        return self.anytask(lib.distance_transform, pixel_pitch=pixel_pitch, background=background, **kwargs)

    def get_faces_with_neighbors(self, **kwargs):
        return self.anytask(lib.get_faces_with_neighbors, **kwargs)

    def gaussian_smoothing(self, sigma, **kwargs):
        return self.anytask(lib.gaussian_smoothing, sigma, **kwargs)

    def extended_local_maxima(self, neighborhood=26, **kwargs):
        return self.anytask(lib.extended_local_maxima, neighborhood=neighborhood, **kwargs)

    def find_bounding_rect(self, **kwargs):
        return self.anytask(lib.find_bounding_rect, **kwargs)

    def crop_bounding_rect(self, bounds=None, **kwargs):
        return self.anytask(lib.crop_bounding_rect, bounds=bounds, **kwargs)

    def mask_image(self, maskvalue=False, value=0, **kwargs):
        return self.anytask(lib.mask_image, maskvalue=maskvalue, value=value, **kwargs)

    def amax(self, return_only=True, **kwargs):
        return self.anytask(lib.amax, return_only=return_only, **kwargs)

    # _________________________________________________________________________________________
    # Iterators

    def label_iterator(self, key=None, labellist=None, background=None, area=None):
        """
        :param key:
        :param labellist:
        :param background:
        :param area: supply area in the format area=np.s_[numpy indexing], i.e. area=np.s_[:,:,:] for a full 3d image
            Note that this affects only the determination of which labels are iterated over, when labellist is supplied
            this parameter has no effect
        :return:
        """

        if labellist is None:
            labellist = self.unique(keys=key, return_only=True)[key]
            if area is not None:

                labellist = lib.unique(self[key][area])

        if background is not None:
            labellist = filter(lambda x: x != 0, labellist)

        for lbl in labellist:
            yield lbl


    def label_image_iterator(self, key=None, labellist=None, background=None,
                             accumulate=False, area=None, relabel=False):

        if accumulate:

            lblim = np.zeros(self[key].shape, dtype=self[key].dtype)

            c = 1
            for lbl in self.label_iterator(key=key, labellist=labellist, background=background, area=area):

                if relabel:
                    lblim[lib.getlabel(self[key], lbl) == 1] = c
                    yield [c, lblim]
                    c += 1
                else:
                    lblim[lib.getlabel(self[key], lbl) == 1] = lbl
                    yield [lbl, lblim]

        else:

            for lbl in self.label_iterator(key=key, labellist=labellist, background=background, area=area):

                lblim = lib.getlabel(self[key], lbl)
                yield [lbl, lblim]

    def label_image_bounds_iterator(self, key=None, labellist=None, background=None,
                                    area=None, more_keys=None,
                                    maskvalue=0, value=0, forcecontinue=False):

        for lbl, lblim in self.label_image_iterator(key=key, background=background, labellist=labellist, area=area):

            # self.logging('self.amax() = {}', self.amax())
            try:

                bounds = lib.find_bounding_rect(lblim, s_=True)
                lblim = lib.crop_bounding_rect(lblim, bounds)

                if more_keys is not None:
                    more_ims = self.crop_bounding_rect(bounds, keys=more_keys, return_only=True)
                    more_ims.mask_image(maskvalue=maskvalue, value=value, keys=more_keys, indict={'lblim': lblim})

                    yield [lbl, lblim, more_ims, bounds]

                else:
                    yield [lbl, lblim, bounds]

            except:

                if forcecontinue:
                    self.errprint('Warning: Something went wrong in label {}, jumping to next label'.format(lbl), traceback)

                    continue
                else:
                    raise


if __name__ == '__main__':

    # hp = Hdf5Processing()
    hipl = Hdf5ImageProcessingLib()

    hipl['a', 'b', 'c1'] = np.zeros((10, 10))
    hipl['a', 'b', 'c2'] = np.ones((10, 10))

    hipl['a', 'b2', 'd'] = np.ones((10, 10))*2
    hipl['a', 'b2', 'e'] = np.ones((10, 10))*3

    hipl['f', 'g', 'h'] = np.ones((10, 10))*5

    print hipl.datastructure2string()
    print '--------------------------'

    hipl.anytask(lib.add2im,
                 keys=(('a', 'b'), ('a', 'b2')),
                 keys2=(('a', 'b'), ('a', 'b2')),
                 reciprocal=True,
                 tkeys='result',
                 return_only=False)
                 #tkeys=zip(('result',)*2, hipl.resultkey_name_gen(('ab', 'ab2'), ('ab', 'ab2'))))

    print hipl.datastructure2string()

    #
    # # hipl.anytask(lib.add2im, keys=(('a', 'b'),), keys2=(('a', 'b2'),), tkeys='result')
    # hipl.anytask(lib.add, 2, tkeys='add2')
    # print hipl.datastructure2string()
    # print '--------------------------'
    #
    # hipl.anytask(lib.add2im, keys=(('a', 'b'),), keys2=(('a', 'b2'),), tkeys='add2im')
    # print hipl.datastructure2string()
    # print '--------------------------'
    # print hipl['a', 'b', 'c2'][0, 0]
    # print hipl['a', 'b2', 'e'][0, 0]
    # print hipl['add2im', 'c2'][0, 0]
    # print '--------------------------'
    #
    # hipl.anytask(lib.add2im, keys=(('a', 'b'), ('a', 'b2')), keys2=(('a', 'b'), ('a', 'b2')), tkeys='add2im_recip', reciprocal=True)
    # print hipl.datastructure2string()
    # print '--------------------------'


    # print type(hipl)
    # print type(hipl['a'])
    # print type(hipl['a','b'])
    # print type(hipl['a','b','c1'])
    #
    # hipl.anytask(lib.add, 10, keys=(('a', 'b'), 'f'))
    # hipl.anytask(lib.mult, 0.5)
    #
    # print hipl['a', 'b', 'c1'][0, 0]
    # print hipl['a', 'b', 'c2'][0, 0]
    # print hipl['f','g','h'][0, 0]
    # print hipl['a', 'b2', 'd'][0, 0]
    # print hipl['a', 'b2', 'e'][0, 0]
    #
    # hipl.anytask(lib.add, 20, keys=(('a', 'b2'), ('f', 'g')), tkeys=(('a', 'b3'), 'f2'))
    #
    # print hipl.datastructure2string()
    # print '---'
    #
    # hipl.anytask(lib.add, 7, tkeys=('result',))
    #
    # print hipl.datastructure2string()
