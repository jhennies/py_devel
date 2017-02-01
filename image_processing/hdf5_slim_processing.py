
from yaml_parameters import YamlParams
import h5py
import numpy as np


class H5pyExtendedFile(h5py.File):

    def __init__(self, *args, **kwargs):

        self._logger = None
        if 'logger' in kwargs.keys():
            self._logger = kwargs.pop('logger')

        h5py.File.__init__(self, *args, **kwargs)

        self.set_data(self)

    def __getitem__(self, item):

        if type(item) is tuple or type(item) is list:

            item = list(item)
            if len(item) > 1:
                first = item.pop(0)
                if len(item) == 1:
                    item = item[0]
                return h5py.File.__getitem__(self, first)[item]

            else:
                return h5py.File.__getitem__(self, item[0])

        else:

            try:
                return h5py.File.__getitem__(self, item)
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
                h5py.File.__setitem__(self, key[0], val)
        else:
            h5py.File.__setitem__(self, key, val)

    # def __add__(self, other):
    #
    #     dictsum = other.dcp()
    #
    #     for d, k, v, kl in self.data_iterator(leaves_only=True):
    #         if dictsum.inkeys(kl):
    #             dictsum[kl] += v
    #         else:
    #             dictsum[kl] = v
    #
    #     return dictsum

    def set_data(self, data):

        for k in data.keys():
            try:
                self[k] = type(self)(data[k])
            except:
                self[k] = data[k]
    #
    # def subset(self, *args, **kwargs):
    #     """
    #     Returnes subset of self
    #     Note that the subset is returned by reference
    #
    #     :param args:
    #     Lets assume self is has the datastructure:
    #         a:
    #             b:
    #                 c: value1
    #                 d: value2
    #             e:
    #                 f: value3
    #         g:
    #             h: value4
    #
    #     if search == False:
    #
    #     subset('a', 'g') would return the full dictionary
    #     subset(('a', 'b')) would return
    #         a:
    #             b:
    #                 c: value1
    #                 d: value2
    #     subset(('a', 'e'), 'g') would return
    #         a:
    #             e:
    #                 f: value3
    #         g:
    #             h: value4
    #
    #     if search == True:
    #
    #     subset('c', 'h', search=True) would return
    #         a:
    #             b:
    #                 c: value1
    #         g:
    #             h: value4
    #
    #     :param search=False:
    #     :return:
    #     """
    #
    #     if 'search' in kwargs.keys():
    #         search = kwargs.pop('search')
    #     else:
    #         search = False
    #
    #     rtrndict = type(self)()
    #     if not search:
    #
    #         for kl in args:
    #             rtrndict[kl] = self[kl]
    #
    #     else:
    #
    #         for d, k, v, kl in self.data_iterator():
    #             if k in args:
    #                 rtrndict[kl] = self[kl]
    #
    #     return rtrndict
    #
    # def merge(self, rdict, overwrite=True):
    #
    #     for d, k, v, kl in rdict.data_iterator(leaves_only=True):
    #         if overwrite:
    #             self[kl] = v
    #         else:
    #             if not self.inkeys(kl):
    #                 self[kl] = v
    #
    # def rename_entry(self, old, new, search=False):
    #     """
    #     Renames an entry, regardless of being node or leaf
    #     :param old:
    #     :param new:
    #     :param search:
    #     :return:
    #     """
    #
    #     if search:
    #         for d, k, v, kl in self.data_iterator(yield_short_kl=True):
    #             if k == old:
    #                 self[kl][new] = self[kl].pop(old)
    #     else:
    #         self[new] = self.pop(old)
    #
    # def datastructure2string(
    #         self, maxdepth=None, data=None, indentstr='.  ', function=None, leaves_only=True,
    #         as_yaml=False
    # ):
    #     if data is None:
    #         data = self
    #
    #     dstr = ''
    #
    #     if as_yaml:
    #
    #         indentstr = '    '
    #
    #         for d, k, v, kl in data.data_iterator(maxdepth=maxdepth):
    #             if type(v) is type(self):
    #                 if type(k) is str:
    #                     dstr += "{}'{}':\n".format(indentstr * d, k)
    #                 else:
    #                     dstr += '{}{}:\n'.format(indentstr * d, k)
    #             else:
    #                 if type(v) is str:
    #                     dstr += "{}{}: '{}'\n".format(indentstr * d, k, v)
    #                 else:
    #                     dstr += '{}{}: {}\n'.format(indentstr * d, k, v)
    #
    #     else:
    #
    #         for d, k, v, kl in data.data_iterator(maxdepth=maxdepth):
    #             if (function is None or type(v) is type(self)) and leaves_only:
    #                 dstr += '{}{}\n'.format(indentstr * d, k)
    #             elif function is None and not leaves_only:
    #                 dstr += '{}{}\n'.format(indentstr * d, k)
    #             else:
    #                 try:
    #                     dstr += '{}{}: {}\n'.format(indentstr * d, k, str(function(v)))
    #                 except:
    #                     dstr += '{}{}\n'.format(indentstr * d, k)
    #
    #     return dstr
    #
    # def dss(self, *args, **kwargs):
    #     """
    #     Just a shorter version of datastructure2string() including the print statement
    #     :param maxdepth:
    #     :param data:
    #     :param indentstr:
    #     :param function:
    #     :return:
    #     """
    #     print_text = 'Dict datastructure: \n---\n{}'
    #     print_value = self.datastructure2string(*args, **kwargs)
    #     if self._logger is not None:
    #         self._logger.logging(print_text, print_value)
    #     else:
    #         print print_text.format(print_value)

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

    # def lengths(self, depth=0):
    #
    #     returndict = type(self)()
    #     for d, k, v, kl in self.data_iterator():
    #         if d == depth:
    #             returndict[kl] = len(v)
    #
    #     return returndict
    #
    # def maxlength(self, depth=0):
    #
    #     maxlen = 0
    #     for d, k, v, kl in self.data_iterator():
    #         if d == depth:
    #             if len(v) > maxlen:
    #                 maxlen = len(v)
    #
    #     return maxlen
    #
    # def maxdepth(self):
    #     maxd = 0
    #     for d, k, v, kl in self.data_iterator():
    #         if d > maxd:
    #             maxd = d
    #     return d
    #
    # def inkeys(self, kl):
    #
    #     if kl:
    #         if kl[0] in self.keys():
    #             if type(self[kl[0]]) is type(self):
    #                 return self[kl[0]].inkeys(kl[1:])
    #             else:
    #                 if len(kl) == 1:
    #                     return True
    #                 elif len(kl) > 1:
    #                     return False
    #                 else:
    #                     raise RuntimeError('Hdf5Processing: This should not have happened!')
    #         else:
    #             return False
    #     else:
    #         return True

    # def dcp(self):
    #     """
    #     Deep copy this instance
    #     """
    #     return type(self)(self)


class RecursiveDict(dict):

    def __init__(self, data=None, logger=None):
        dict.__init__(self)

        self._logger = logger

        if data is not None:
            self.set_data(data)

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

    def set_data(self, data):

        for k in data.keys():
            try:
                self[k] = type(self)(data[k])
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
            self, maxdepth=None, data=None, indentstr='.  ', function=None, leaves_only=True,
            as_yaml=False
    ):
        if data is None:
            data = self

        dstr = ''

        if as_yaml:

            indentstr = '    '

            for d, k, v, kl in data.data_iterator(maxdepth=maxdepth):
                if type(v) is type(self):
                    if type(k) is str:
                        dstr += "{}'{}':\n".format(indentstr * d, k)
                    else:
                        dstr += '{}{}:\n'.format(indentstr * d, k)
                else:
                    if type(v) is str:
                        dstr += "{}{}: '{}'\n".format(indentstr * d, k, v)
                    else:
                        dstr += '{}{}: {}\n'.format(indentstr * d, k, v)

        else:

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

    def dss(self, *args, **kwargs):
        """
        Just a shorter version of datastructure2string() including the print statement
        :param maxdepth:
        :param data:
        :param indentstr:
        :param function:
        :return:
        """
        print_text = 'Dict datastructure: \n---\n{}'
        print_value = self.datastructure2string(*args, **kwargs)
        if self._logger is not None:
            self._logger.logging(print_text, print_value)
        else:
            print print_text.format(print_value)

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

    def dcp(self):
        """
        Deep copy this instance
        """
        return type(self)(self)


class Hdf5Processing(RecursiveDict):

    def __init__(
            self, data=None, logger=None,
            filepath=None, skeys=None, tkeys=None, recursive_search=False, nodata=False
    ):

        self._sources = None
        self._f = None

        if data is not None:
            RecursiveDict.__init__(self, data, logger=logger)
        else:
            RecursiveDict.__init__(self, logger=logger)

        if filepath is not None:
            self.data_from_file(filepath, skeys, tkeys, recursive_search, nodata)

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
            self.set_data(f)
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
        newentries.set_data(f)

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

    def data_from_file(
            self, filepath, skeys=None, tkeys=None, recursive_search=False, nodata=False
    ):

        self._f = h5py.File(filepath)
        self.get_h5_entries(self._f, skeys=skeys, tkeys=tkeys, recursive_search=recursive_search)

        if not nodata:
            self.populate()

    def close(self):
        self._f.close()


if __name__ == '__main__':

    filepath = '/mnt/localdata02/jhennies/neuraldata/results/cremi_2016/170130_slice_selection_test_z_develop/intermed/cremi.splA.train.segmlarge.crop.crop_x10_110_y200_712_z200_712.split_z.h5'

    a = Hdf5Processing(
        filepath=filepath, nodata=True
    )


    a.dss(as_yaml=True)

    # f = H5pyExtendedFile(filepath)

    # f.dss()

    pass