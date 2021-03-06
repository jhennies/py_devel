
import numpy as np
import h5py
from simple_logger import SimpleLogger
from yaml_parameters import YamlParams
import multiprocessing
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

__author__ = 'jhennies'


class RecursiveDict(dict, SimpleLogger):

    def __init__(self, data=None, tkeys=None):
        dict.__init__(self)

        if data is not None:
            if tkeys is None:
                self.setdata(data)
            else:
                self.setdata(data, tkeys)

    def __getitem__(self, item):

        if type(item) is tuple or type(item) is list:

            item = list(item)
            if len(item) > 1:
                first = item.pop(0)
                return dict.__getitem__(self, first)[item]

            else:
                return dict.__getitem__(self, item[0])

        else:

            try:
                return dict.__getitem__(self, item)
            except KeyError:
                value = self[item] = type(self)()
                return value

    def __setitem__(self, key, val):
        if type(key) is tuple or type(key) is list:
            if len(key) > 1:
                key = list(key)
                fkey = key.pop(0)
                self[fkey][key] = val
            else:
                dict.__setitem__(self, key[0], val)
        else:
            dict.__setitem__(self, key, val)

    def __add__(self, other):

        dictsum = other.dcp()

        for d, k, v, kl in self.data_iterator(leaves_only=True):
            if dictsum.inkeys(kl):
                dictsum[kl] += v
            else:
                dictsum[kl] = v

        return dictsum

    def setdata(self, data):

        for k in data.keys():
            try:
                self[k] = type(self)(data=data[k])
            except:
                self[k] = data[k]

    def subset(self, *args, **kwargs):
        """
        Returnes subset of self
        Note that the subset is returned by reference

        :param args:
        Lets assume self is has the datastructure:
            a:
                b:
                    c: value1
                    d: value2
                e:
                    f: value3
            g:
                h: value4

        if search == False:

        subset('a', 'g') would return the full dictionary
        subset(('a', 'b')) would return
            a:
                b:
                    c: value1
                    d: value2
        subset(('a', 'e'), 'g') would return
            a:
                e:
                    f: value3
            g:
                h: value4

        if search == True:

        subset('c', 'h', search=True) would return
            a:
                b:
                    c: value1
            g:
                h: value4

        :param search=False:
        :return:
        """

        if 'search' in kwargs.keys():
            search = kwargs.pop('search')
        else:
            search = False

        rtrndict = type(self)()
        if not search:

            for kl in args:
                rtrndict[kl] = self[kl]

        else:

            for d, k, v, kl in self.data_iterator():
                if k in args:
                    rtrndict[kl] = self[kl]

        return rtrndict

    def merge(self, rdict, overwrite=True):

        for d, k, v, kl in rdict.data_iterator(leaves_only=True):
            if overwrite:
                self[kl] = v
            else:
                if not self.inkeys(kl):
                    self[kl] = v

    def rename_entry(self, old, new, search=False):
        """
        Renames an entry, regardless of being node or leaf
        :param old:
        :param new:
        :param search:
        :return:
        """

        if search:
            for d, k, v, kl in self.data_iterator(yield_short_kl=True):
                if k == old:
                    self[kl][new] = self[kl].pop(old)
        else:
            self[new] = self.pop(old)

    def datastructure2string(
            self, maxdepth=None, data=None, indentstr='.  ', function=None,
            leaves_only=True
    ):
        if data is None:
            data = self

        dstr = ''

        for d, k, v, kl in data.data_iterator(maxdepth=maxdepth):
            if (function is None or type(v) is type(self)) and leaves_only:
                dstr += '{}{}\n'.format(indentstr * d, k)
            elif function is None and not leaves_only:
                dstr += '{}{}\n'.format(indentstr * d, k)
            else:
                try:
                    dstr += '{}{}: {}\n'.format(indentstr * d, k, str(function(v)))
                except:
                    dstr += '{}{}\n'.format(indentstr * d, k)

        return dstr

    def dss(self, maxdepth=None, data=None, indentstr='.  ', function=None):
        """
        Just a shorter version of datastructure2string() including the print statement
        :param maxdepth:
        :param data:
        :param indentstr:
        :param function:
        :return:
        """
        self.logging('Dict datastructure: \n---\n{}',
            self.datastructure2string(maxdepth=maxdepth, data=data, indentstr=indentstr,
                                      function=function)
        )

    def data_iterator(
            self, maxdepth=None, data=None, depth=0, keylist=[], yield_short_kl=False,
            leaves_only=False, branches_only=False
    ):

        depth += 1
        if maxdepth is not None:
            if depth-1 > maxdepth:
                return

        if data is None:
            data = self

        try:

            for key, val in data.iteritems():

                if not yield_short_kl:
                    kl = keylist + [key,]
                else:
                    kl = keylist

                if not leaves_only and not branches_only:
                    yield [depth-1, key, val, kl]
                if leaves_only:
                    if maxdepth is None:
                        if type(val) is not type(self):
                            yield [depth-1, key, val, kl]
                    else:
                        if type(val) is not type(self) or depth == maxdepth:
                            yield [depth-1, key, val, kl]
                if branches_only:
                    if type(val) is type(self):
                        yield [depth-1, key, val, kl]

                if yield_short_kl:
                    kl = keylist + [key,]

                for d in self.data_iterator(
                    maxdepth=maxdepth, data=val, depth=depth, keylist=kl,
                    yield_short_kl=yield_short_kl, leaves_only=leaves_only,
                    branches_only=branches_only
                ):
                    yield d

        except AttributeError:
            pass

    def simultaneous_iterator(self, data=None, keylist=None, max_count_per_item=None):

        if data is None:
            data = self

        if max_count_per_item is None:
            max_count_per_item = data.maxlength(depth=0)
        else:
            if max_count_per_item > data.maxlength(depth=0):
                max_count_per_item = data.maxlength(depth=0)

        if keylist is None:

            for i in xrange(max_count_per_item):
                keys = []
                vals = []
                for key, val in data.iteritems():
                    try:
                        vals.append(val[val.keys()[i]])
                        keys.append((key, val.keys()[i]))
                    except:
                        pass
                yield [i, keys, vals]

        else:

            for key in keylist:
                keys = []
                vals = []
                for dkey, dval in data.iteritems():
                    try:
                        if key in dval.keys():
                            vals.append(dval[key])
                            keys.append((dkey, key))
                    except:
                        pass
                yield [key, keys, vals]

    def multiprocessing(
            self, myfunction, n=4, maxdepth=None, yield_short_kl=False, leaves_only=False,
            branches_only=False, add=None
    ):
        """
        :param myfunction: The function to be executed
        :param n: number of threads to use

        For more kwargs refer to data_iterator

        EXAMPLE

        # Initialize and load data
        data = Hdf5Processing()
        data.data_from_file(filename='/path/to/data.h5')

        # Define the function which will be multithreaded
        def data_processing(p):
            # Use only one input parameter which contains the following data (see data_iterator)
            d = p[0]
            k = p[1]
            kl = p[2]

            # Some computation with a dataitem obtained by use of kl
            # data has to be available here from scope, i.e. no passing as argument possible
            do_something_with_data_item(data[kl])

        # This is the multiprocessing call
        data.multiprocessing(data_processing, n=4, leaves_only=True)

        """

        # Prepare input array for multiprocessing using data_iterator and storing depth (d),
        #   the current key (k) and keylist (kl) into an array. Note that the data (v) is not
        #   stored and can only be available in myfunction if it is found in its respective
        #   scope. (See code example above)
        p_array = []
        for d, k, v, kl in self.data_iterator(
                maxdepth=maxdepth, yield_short_kl=yield_short_kl, leaves_only=leaves_only,
                branches_only=branches_only
        ):
            if add is None:
                p_array.append([d, k, kl])
            else:
                p_array.append([d, k, kl, add])

        # Create processing pool and run
        pool = multiprocessing.Pool(n)
        pool.map(myfunction, p_array)

    def multiprocessing_split(
            self, myfunction, n=4, maxdepth=None, yield_short_kl=False, leaves_only=False,
            branches_only=False, add=None
    ):

        # Prepare input array for multiprocessing using data_iterator and storing depth (d),
        #   the current key (k) and keylist (kl) into an array. Note that the data (v) is not
        #   stored and can only be available in myfunction if it is found in its respective
        #   scope. (See code example above)
        p_array = []
        c = 0
        for d, k, v, kl in self.data_iterator(
                maxdepth=maxdepth, yield_short_kl=yield_short_kl, leaves_only=leaves_only,
                branches_only=branches_only
        ):
            if add is None:
                p_array.append([d, k, np.array(v), kl])
            else:
                p_array.append([d, k, np.array(v), kl, add])

            if c/float(n) == c/n:
                # Create processing pool and run
                pool = multiprocessing.Pool(n)
                pool.map(myfunction, p_array)
                p_array = []

            c += 1

    def lengths(self, depth=0):

        returndict = type(self)()
        for d, k, v, kl in self.data_iterator():
            if d == depth:
                returndict[kl] = len(v)

        return returndict

    def maxlength(self, depth=0):

        maxlen = 0
        for d, k, v, kl in self.data_iterator():
            if d == depth:
                if len(v) > maxlen:
                    maxlen = len(v)

        return maxlen

    def maxdepth(self):
        maxd = 0
        for d, k, v, kl in self.data_iterator():
            if d > maxd:
                maxd = d
        return d

    def switch_levels(self, level1, level2):
        newself = type(self)()
        for d, k, v, kl in self.data_iterator(maxdepth=level2):
            if d == level2:
                newkl = list(kl)
                newkl[level1] = kl[level2]
                newkl[level2] = kl[level1]

                newself[newkl] = self[kl]

        return newself

    def remove_layer(self, layername):

        for d, k, v, kl in self.data_iterator(yield_short_kl=True):
            if k == layername and type(v) is type(self):
                # tself = type(self)(data=self[kl + [k]])
                # del self[kl][k]
                # self[kl] = tself
                self[kl] = self[kl].pop(layername)

    def delete_items(self, key):

        for d, k, v, kl in self.data_iterator(yield_short_kl=True):
            if k == key:
                del self[kl][k]

    def inkeys(self, kl):

        if kl:
            if kl[0] in self.keys():
                if type(self[kl[0]]) is type(self):
                    return self[kl[0]].inkeys(kl[1:])
                else:
                    if len(kl) == 1:
                        return True
                    elif len(kl) > 1:
                        return False
                    else:
                        raise RuntimeError('Hdf5Processing: This should not have happened!')
            else:
                return False
        else:
            return True

    def rename_layer(self, layername, layernewname):

        for d, k, v, kl in self.data_iterator(yield_short_kl=True):

            if self.inkeys(kl + [k]):
                if k == layername and type(v) is type(self):

                    if kl:

                        # if len(self[kl].keys()) == 1:

                        # This is really ugly but somehow popping won't work
                        self[kl + [layernewname]] = type(self)(data=self[kl + [k]])
                        t = self[kl]
                        del t[k]
                        self[kl] = t

                        # else:
                        #     self[kl + [layernewname]] = self[kl].pop(layername)
                        # print self[kl].datastructure2string(maxdepth=2)
                    else:
                        # print self.datastructure2string(maxdepth=2)
                        # print type(self)
                        self[layernewname] = self.pop(layername)
                        # print self.datastructure2string(maxdepth=2)

    def reduce_from_leaves(self, iterate=False):
        """
        If the branches at the leaves have only one entry, i.e.
        a:
            b:
                c: leaf1
            d:
                e: leaf2
        f:
            g:
                h: leaf3
        the last level is removed resulting in:
        a:
            b: leaf1
            d: leaf2
        f:
            g: leaf3

        A further iteration would yield:
        a:
            b: leaf1
            d: leaf2
        f: leaf3

        :return:
        """

        if not iterate:
            for d, k, v, kl in self.data_iterator(yield_short_kl=True):

                if type(v) is not type(self):
                    if kl:
                        if len(self[kl].keys()) == 1:
                            self[kl] = v

        else:

            busy = True
            while busy:

                busy = False
                for d, k, v, kl in self.data_iterator(yield_short_kl=True, leaves_only=True):

                    if kl:
                        if len(self[kl].keys()) == 1:
                            self[kl] = v
                            busy = True

    def dcp(self):
        """
        Deep copy this instance
        """
        return type(self)(data = self)

    def yield_an_item(self):
        for d, k, v, kl in self.data_iterator():
            if type(v) is not type(self):
                return v


class Hdf5Processing(RecursiveDict, YamlParams):

    _sources = None

    def __init__(self, path=None, filename=None, filepath=None, data=None, skeys=None, tkeys=None, castkey=None,
                 yaml=None, yamlspec=None, recursive_search=False, nodata=False):

        if type(yaml) is str:
            YamlParams.__init__(self, filename=yaml)
        elif isinstance(yaml, YamlParams):
            YamlParams.__init__(self, yaml=yaml)

        # self._sources = dict()

        if yaml is not None:

            yamldict = self.get_params()
            if yamlspec is None:
                if 'filepath' in yamldict.keys(): filepath = yamldict['filepath']
                if 'path' in yamldict.keys(): filepath = yamldict['path']
                if 'filename' in yamldict.keys(): filepath = yamldict['filename']
                if 'castkey' in yamldict.keys(): castkey = yamldict['castkey']
                if 'skeys' in yamldict.keys():
                    if type(yamldict['skeys']) is dict:
                        skeys = ()
                        for i in xrange(1, len(yamldict['skeys'])):
                            skeys += (yamldict[yamldict['skeys'][0]][yamldict['skeys'][i]],)
                    else:
                        skeys = yamldict['skeys']

            else:
                if 'filepath' in yamlspec.keys(): filepath = yamldict[yamlspec['filepath']]
                if 'path' in yamlspec.keys(): path = yamldict[yamlspec['path']]
                if 'filename' in yamlspec.keys(): filename = yamldict[yamlspec['filename']]
                if 'castkey' in yamlspec.keys(): castkey = yamldict[yamlspec['castkey']]
                if 'skeys' in yamlspec.keys():
                    if type(yamlspec['skeys']) is dict:
                        skeys = ()
                        for key, val in yamlspec['skeys'].iteritems():
                            for i in val:
                                skeys += (yamldict[key][i],)

                        for i in xrange(1, len(yamlspec['skeys'])):
                            skeys += (yamldict[yamlspec['skeys'][0]][yamlspec['skeys'][i]],)
                    else:
                        skeys = yamldict[yamlspec['skeys']]



        if data is not None:
            RecursiveDict.__init__(self, data=data, tkeys=tkeys)
        else:
            RecursiveDict.__init__(self)

            if path is not None:
                self.data_from_file(path+filename, skeys=skeys, tkeys=tkeys, castkey=castkey,
                                    recursive_search=recursive_search, nodata=nodata)

            elif filepath is not None:
                self.data_from_file(filepath, skeys=skeys, tkeys=tkeys, castkey=castkey,
                                    recursive_search=recursive_search, nodata=nodata)

    def set_f(self, f):
        self._f = f

    # def write_dataitem(self, group, keys):
    #
    #     if type(v) is type(self):
    #         grp = group.create_group(str(k))
    #         self.write_dataitem(grp, v)
    #     else:
    #         try:
    #             group.create_dataset(k, data=v)
    #         except AttributeError:
    #             group.create_dataset(str(k), data=v)

    def write(self, filepath=None, data=None, of=None, keys=None, search=False):
        """
        :param filepath: Full path to targetfile
        :param data: Data to write (type = Hdf5Instance)
        :param of: Filestream to targetfile
        :param keys: Tuple of parameters for function subset()
        :param search: See documenation in function subset()
        :return:
        """

        if data is None:
            data = self

        if of is None:
            if filepath is not None:
                of = h5py.File(filepath)
            else:
                raise RuntimeError('Hdf5Processing: Error: Specify either [of] or [filepath]!')

        if keys is None:
            data.write_dataset(of, self)
            of.close()
        else:
            data.subset(*keys, search=search).write(filepath=filepath, of=of)

    def write_dataset(self, group, data):

        for k, v in data.iteritems():

            if type(v) is type(self):

                try:
                    grp = group.create_group(str(k))
                except ValueError:
                    grp = group.get(str(k))

                self.write_dataset(grp, v)

            else:

                try:
                    group.create_dataset(k, data=v)
                except AttributeError:
                    group.create_dataset(str(k), data=v)
    #
    # def write(self, filepath=None, of=None):
    #
    #     if of is None:
    #         of = h5py.File(filepath)
    #
    #     self.write_dataset(of, self)
    #
    #     of.close()

    def set_source(self, source, key):
        try:
            self._sources[key] = source
        except TypeError:
            self._sources = type(self)()
            self._sources[key] = source

    def get_sources(self):
        return self._sources

    def clear_sources(self):
        for d, k, v, kl in self.data_iterator(yield_short_kl=True, leaves_only=True):
            self[kl].set_source(None, k)

    def populate(self, key=None):

        if key is None:
            for d, k, v, kl in self.data_iterator(yield_short_kl=True):

                if kl:
                    if type(self[kl][k]) is h5py.Dataset:
                        self[kl].set_source(v, k)
                        self[kl][k] = np.array(v)
                else:
                    if type(self[k]) is h5py.Dataset:
                        self.set_source(v, k)
                        self[k] = np.array(v)

        else:

            try:
                self[key].populate()
            except AttributeError:
                if type(self[key]) is h5py.Dataset:
                    if type(key) is list or type(key) is tuple:
                        lastkey = key.pop(-1)
                        self[key].set_source(self[key][lastkey], lastkey)
                        self[key][lastkey] = np.array(self[key][lastkey])
                    else:
                        self.set_source(self[key], key)
                        self[key] = np.array(self[key])
                else:
                    raise

    def unpopulate(self, key=None):

        if key is None:
            for d, k, v, kl in self.data_iterator(yield_short_kl=True):

                if kl:
                    if self[kl].get_sources() is not None:
                        if type(self[kl].get_sources()[k]) is h5py.Dataset:
                            self[kl + [k]] = self[kl].get_sources()[k]
                else:
                    if self.get_sources() is not None:
                        if type(self.get_sources()[k]) is h5py.Dataset:
                            self[k] = self.get_sources()[k]

        else:

            try:
                self[key].unpopulate()
            except AttributeError:
                lastkey = key.pop(-1)
                if type(self[key].get_sources()[lastkey]) is h5py.Dataset:
                    self[key][lastkey] = self[key].get_sources()[lastkey]

    def get_h5_entries(self, f, skeys=None, tkeys=None, recursive_search=False):

        if skeys is None:

            if recursive_search:
                raise TypeError('hdf5_processing.Hdf5Processing.get_h5_entries: skeys must be specified when recursive_search set to True!')

            # We are done here (every entry in the file is set to the dictionary)
            self.setdata(f)
            return

        # From here on, skeys is not None ...

        if type(skeys) is str:
            if recursive_search:
                skeys = (skeys,)
            else:
                skeys=[[skeys]]
        if tkeys is None:
            tkeys = skeys
        elif type(tkeys) is str:
            tkeys = (tkeys,)

        # tkeys is now also not None and both, skeys and tkeys, are tuples

        # Simply set every entry in a temporary dictionary (since we are not actually loading stuff here)
        newentries = type(self)()
        newentries.setdata(f)

        if recursive_search:

            if len(skeys) == 1:

                for d, k, v, kl in newentries.data_iterator(yield_short_kl=True):

                    if k in skeys:

                        keyid = skeys.index(k)
                        tkey = tkeys[keyid]

                        self[kl + [tkey]] = v

            else:

                for d, k, v, kl in newentries.data_iterator():

                    if len(kl) >= len(skeys):

                        if np.array_equal(np.array(kl)[-len(skeys):], np.array(skeys)):

                            self[kl] = v

        else:

            # When not using recursive_search wild cards are now allowed

            def find(a, b):

                def items_equal(item0, item1):
                    if len(item0) != len(item1):
                        return False
                    else:
                        for i in xrange(0, len(item0)):
                            if item0[i] != item1[i] and item0[i] != '*' and item1[i] != '*':
                                return False
                        return True

                # Find a in b and return True or False
                for bi in b:
                    if items_equal(a, bi):
                        return True

                return False

            for d, k, v, kl in newentries.data_iterator(yield_short_kl=True):

                # if kl + [k] in skeys:
                if find(kl + [k], skeys):

                    try:
                        keyid = skeys.index(kl + [k])
                        tkey = tkeys[keyid]
                        if type(tkey) is str:
                            tkey = [tkey]

                        self[tkey] = v

                    except ValueError:
                        self[kl + [k]] = v

    def data_from_file(self, filepath, skeys=None, tkeys=None, castkey=None,
                       recursive_search=False, nodata=False):

        f = h5py.File(filepath)
        self.get_h5_entries(f, skeys=skeys, tkeys=tkeys, recursive_search=recursive_search)

        if not nodata:
            self.populate()


if __name__ == '__main__':

    a = Hdf5Processing()
    a.data_from_file(
        filepath='/mnt/localdata02/jhennies/neuraldata/results/cremi_2016/170109_neurobioseg_x_cropped_add_featureims_develop/intermed/cremi.splA.train.borderct.crop.crop_x10_110_y200_712_z200_712.split_x.h5',
        nodata=True,
        skeys=[['x', '0', '*', 'contacts']]
    )
    a.dss(maxdepth=4)

    # a = RecursiveDict(data={'a': {'b': {'c': 1, 'd': 2}, 'e': {'f': 3}}, 'g': {'h': {'i': 4}}})
    # a.dss()
    # b = a.subset('c', 'h', search=True)

    # resultfolder = '/mnt/localdata02/jhennies/neuraldata/results/cremi_2016/161111_random_forest_of_paths_add_features_develop/'
    #
    # yamlfile = resultfolder + '/parameters.yml'
    #
    # ipl = Hdf5Processing(
    #     yaml=yamlfile,
    #     yamlspec={'path': 'intermedfolder'},
    #     filename='test.h5',
    #     skeys=('disttransf', 'disttransfm'),
    #     recursive_search=True,
    #     nodata=True
    # )
    # ipl.dss()
    #
    # def populated(data):
    #     if type(data) is h5py.Dataset:
    #         return 'false'
    #     else:
    #         return 'TRUE'
    #
    # # def firstvalue(data):
    # #     return data[0, 0, 0]
    # #
    # # print ipl.dss(function=populated)
    # #
    # # ipl.populate()
    # # ipl.unpopulate(['x', '0', 'raw'])
    #
    # ipl.dss(function=populated)
    #
    # ipl.write(filepath=ipl.get_params()['intermedfolder'] + 'test2.h5')

    pass

    # hfp = Hdf5Processing()
    # content = hfp.load('/media/julian/Daten/neuraldata/cremi_2016/develop/161011_locmax_paths_feature_extraction/intermediate/cremi.splA.raw_neurons.crop.crop_10-200-200_110-712-712.paths.true.h5',
    #          castkey=float)
    # print content.keys()
    # print content[7428].keys()
    # print content[7428][0

    # hfp = Hdf5Processing(
    #     filepath='/media/julian/Daten/neuraldata/cremi_2016/develop/161011_locmax_paths_feature_extraction/intermediate/cremi.splA.raw_neurons.crop.crop_10-200-200_110-712-712.paths.true.h5',
    #     dataname='content',
    #     castkey=float
    # )
    #
    # data = hfp['content']

    # print data.keys()
    # print data[data.keys()[0]].keys()
    # print data[data.keys()[0]][data[data.keys()[0]].keys()[0]]

    # yamlfile = '/media/julian/Daten/src/hci/py_devel/neurobioseg/161005_locmax_paths_feature_extraction/parameters.yml'
    #
    # hfp = Hdf5Processing(
    #     yaml=yamlfile,
    #     yamlspec={'path': 'intermedfolder', 'filename': 'pathstruefile'},
    #     dataname='true',
    #     castkey=None
    # )
    # params = hfp.get_params()
    # hfp.logging('params = {}', params)
    # hfp.data_from_file(
    #     filepath=params['intermedfolder'] + params['pathsfalsefile'],
    #     dataname='false',
    #     castkey=None
    # )
    # hfp.startlogger()

    # hfp.logging('Datastructure:\n{}', hfp.datastructure2string())
    # for i in hfp.data_iterator():
    #     print i['keylist']

    # hfp['true']['100']['39'] = 10
    # hfp['true', '100', '39'] = 10
    # hfp.logging('{}', hfp['true'].keys())
    # hfp.logging('{}', hfp['true', '12735.0', '39'])
    #
    # hfp.logging('{}', hfp['true', '12735.0'].keys())
    # hfp.logging('{}', hfp['true', '12735.0', '11'])

    # print hfp['true']['12735.0']

    # print type(hfp['true', '12735.0'])
    #
    # a = ('test', 'a')
    # # hfp['test', 'a'] = 12
    # hfp[a] = 12
    # print hfp.keys()
    # print hfp['test'].keys()
    #
    # a = ('false', 'a', '70')
    # # hfp[a] = 99
    # hfp['false', 'a', '70'] = 99
    # print hfp['false', 'a'].keys()

    # hfp.stoplogger()

    #
    # hfp = Hdf5Processing(data={'a': {'b': np.array([10, 10])}})
    #
    # hfp['a', 'c'] = np.array([20, 10, 20, 30])
    # hfp['d', 'b'] = [[1, 1, 2], [2, 3, 4], [3, 2, 1], [2], [12, 3], [4, 3,4]]
    #
    # print hfp
    # print hfp.datastructure2string()
    #
    # hfp.write(filepath='/home/julian/Documents/test.h5')