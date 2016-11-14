import h5py
import numpy as np
from simple_logger import SimpleLogger
from yaml_parameters import YamlParams
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


class Hdf5Processing(dict, YamlParams):

    _sources = None

    def __init__(self, path=None, filename=None, filepath=None, data=None, skeys=None, tkeys=None, castkey=None,
                 yaml=None, yamlspec=None, recursive_search=False, nodata=False):

        dict.__init__(self)
        YamlParams.__init__(self, filename=yaml)

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
            if tkeys is None:
                self.setdata(data)
            else:
                self.setdata(data, tkeys)

        elif path is not None:
            self.data_from_file(path+filename, skeys=skeys, tkeys=tkeys, castkey=castkey,
                                recursive_search=recursive_search, integrate=True, nodata=nodata)

        elif filepath is not None:
            self.data_from_file(filepath, skeys=skeys, tkeys=tkeys, castkey=castkey,
                                recursive_search=recursive_search, integrate=True, nodata=nodata)

    def __getitem__(self, items):

        if type(items) is tuple or type(items) is list:

            items = list(items)
            if len(items) > 1:
                firstitem = items.pop(0)
                return dict.__getitem__(self, firstitem)[items]

            else:
                return dict.__getitem__(self, items[0])

        else:

            try:
                return dict.__getitem__(self, items)
            except KeyError:
                value = self[items] = type(self)()
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

    def setdata(self, data, tkeys=None):

        for k in data.keys():
            try:
                self[k] = type(self)(data=data[k])
            except:
                self[k] = data[k]

    def write_dataset(self, group, data):

        for k, v in data.iteritems():

            if type(v) is type(self):

                grp = group.create_group(str(k))

                self.write_dataset(grp, v)

            else:

                if type(v) is list:

                    grp = group.create_group(str(k))
                    for i in xrange(0, len(v)):
                        grp.create_dataset(str(i), data=v[i])

                elif type(v) is np.ndarray:
                    try:
                        group.create_dataset(k, data=v)
                    except AttributeError:
                        group.create_dataset(str(k), data=v)

                else:
                    print 'Warning in Hdf5Processing.write(): Nothing to write.'

    def write(self, filepath=None, of=None):

        if of is None:
            of = h5py.File(filepath)

        self.write_dataset(of, self)

        of.close()

    def get_h5_content(self, f, skeys=None, tkeys=None, offset='    ', castkey=None,
                       recursive_search=False, integrate=False, selfinstance=None, nodata=False):

        # if isinstance(f, h5py.Dataset):
        #     print offset, '(Dataset)', f.name, 'len =', f.shape
        #
        # if isinstance(f, h5py.Group):
        #     print offset, '(Group)', f.name
        #
        # else :
        #     pass
        #     # print 'WARNING: UNKNOWN ITEM IN HDF5 FILE', f.name
        #     # sys.exit ( "EXECUTION IS TERMINATED" )
        # dict_f = {}

        if recursive_search:

            if integrate:

                if isinstance(f, h5py.File) or isinstance(f, h5py.Group):
                    if selfinstance is None:
                        selfinstance = self
                    if skeys is None:
                        raise TypeError('hdf5_processing.Hdf5Processing.get_h5_content: skeys must be specified when recursive_search set to True!')
                    if type(skeys) is str:
                        skeys = (skeys,)

                    rtrn_dict = selfinstance
                    dict_f = dict(f)
                    for key, val in dict_f.iteritems():
                        subg = val
                        if isinstance(subg, h5py.File) or isinstance(subg, h5py.Group) or key in skeys:
                            if key in skeys:
                                content = self.get_h5_content(subg, nodata=nodata)
                            else:

                                content = self.get_h5_content(subg, skeys=skeys,
                                                              recursive_search=True,
                                                              integrate=True,
                                                              selfinstance=type(self)(data=rtrn_dict)[key], nodata=nodata)
                            if type(content) is type(self):
                                if content:
                                    rtrn_dict[key] = content
                            else:
                                rtrn_dict[key] = content
                else:
                    rtrn_dict = np.array(f)

                return rtrn_dict

            else:

                if isinstance(f, h5py.File) or isinstance(f, h5py.Group):
                    if skeys is None:
                        raise TypeError('hdf5_processing.Hdf5Processing.get_h5_content: skeys must be specified when recursive_search set to True!')
                    if type(skeys) is str:
                        skeys = (skeys,)
                    rtrn_dict = {}
                    dict_f = dict(f)
                    for key, val in dict_f.iteritems():
                        subg = val
                        if isinstance(subg, h5py.File) or isinstance(subg, h5py.Group) or key in skeys:
                            content = self.get_h5_content(subg, skeys=skeys, recursive_search=True, nodata=nodata)
                            if type(content) is dict:
                                if content:
                                    rtrn_dict[key] = content
                            else:
                                rtrn_dict[key] = content
                else:
                    rtrn_dict = np.array(f)

                return rtrn_dict

        else:

            if isinstance(f, h5py.File) or isinstance(f, h5py.Group):
                dict_f = dict(f)
                if skeys is None:
                    skeys = dict_f.keys()
                if type(skeys) is str:
                    skeys = (skeys,)
                if tkeys is None:
                    tkeys = skeys
                if type(tkeys) is str:
                    tkeys = (tkeys,)
                akeys = dict(zip(skeys, tkeys))
                rtrn_dict = {}
                for key, val in dict_f.iteritems():
                    subg = val
                    # print offset, key
                    if castkey is not None:
                        key = castkey(key)
                        # print '{} type(key) = {}'.format(offset, type(key))
                    if key in skeys:
                        rtrn_dict[akeys[key]] = self.get_h5_content(subg, offset=offset + '    ', castkey=castkey, nodata=nodata)

            else:
                # print offset, f
                rtrn_dict = np.array(f)

            return rtrn_dict

    def set_source(self, source, key):
        try:
            self._sources[key] = source
        except TypeError:
            self._sources = type(self)()
            self._sources[key] = source

    def get_sources(self):
        return self._sources

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
                    lastkey = key.pop(-1)
                    self[key].set_source(self[key][lastkey], lastkey)
                    self[key][lastkey] = np.array(self[key][lastkey])
                else:
                    raise

    def unpopulate(self):

        for d, k, v, kl in self.data_iterator(yield_short_kl=True):

            if kl:
                if self[kl].get_sources() is not None:
                    if type(self[kl].get_sources()[k]) is h5py.Dataset:
                        self[kl + [k]] = self[kl].get_sources()[k]
            else:
                if self.get_sources() is not None:
                    if type(self.get_sources()[k]) is h5py.Dataset:
                        self[k] = self.get_sources()[k]

    def get_h5_entries(self, f, skeys=None, tkeys=None, recursive_search=False):

        if skeys is None:

            if recursive_search:
                raise TypeError('hdf5_processing.Hdf5Processing.get_h5_entries: skeys must be specified when recursive_search set to True!')

            # We are done here (every entry in the file is set to the dictionary)
            self.setdata(f)
            return

        # From here on, skeys is not None ...

        if type(skeys) is str:
            skeys=(skeys,)
        if tkeys is None:
            tkeys = skeys
        elif type(tkeys) is str:
            tkeys = (tkeys,)

        # tkeys is now also not None and both, skeys and tkeys, are tuples

        # Simply set every entry in a temporary dictionary (since we are not actually loading stuff here)
        newentries = type(self)()
        newentries.setdata(f)

        if recursive_search:

            for d, k, v, kl in newentries.data_iterator(yield_short_kl=True):

                if k in skeys:

                    keyid = skeys.index(k)
                    tkey = tkeys[keyid]

                    self[kl + [tkey]] = v

        else:

            for d, k, v, kl in newentries.data_iterator(yield_short_kl=True):

                if kl + [k] in skeys:

                    keyid = skeys.index(kl + [k])
                    tkey = tkeys[keyid]
                    if type(tkey) is str:
                        tkey = [tkey]

                    self[tkey] = v


    def load_h5(self, filepath, skeys=None, tkeys=None, castkey=None, recursive_search=False,
                integrate=False, nodata=False):

        f = h5py.File(filepath)

        self.get_h5_entries(f, skeys=skeys, tkeys=tkeys, recursive_search=recursive_search)

    def data_from_file(self, filepath, skeys=None, tkeys=None, castkey=None,
                       recursive_search=False, integrate=False, nodata=False):
        self.load_h5(filepath, skeys=skeys, tkeys=tkeys, castkey=castkey,
                     recursive_search=recursive_search, integrate=integrate, nodata=nodata)
        if not nodata:
            self.populate()

    def getdataitem(self, itemkey):
        return self[itemkey]

    def datastructure2string(self, data=None, dstr='', indent=0, maxdepth=None, depth=0,
                             indentstr='.  '):

        depth += 1
        if maxdepth is not None:
            if depth > maxdepth:
                return dstr

        if data is None:
            data = self

        # if type(data) is dict:

        try:

            for key, val in data.iteritems():

                # print key
                dstr += '{}{}\n'.format(indentstr*indent, key)
                dstr = self.datastructure2string(data=val, dstr=dstr, indent=indent+1, maxdepth=maxdepth, depth=depth, indentstr=indentstr)

        except:
            pass
        return dstr

    def data_iterator(self, maxdepth=None, data=None, depth=0, keylist=[], yield_short_kl=False):

        depth += 1
        if maxdepth is not None:
            if depth-1 > maxdepth:
                return

        if data is None:
            data = self

        try:

            for key, val in data.iteritems():
                # print key, val
                if yield_short_kl:
                    yield [depth-1, key, val, keylist]
                    kl = keylist + [key,]
                else:
                    kl = keylist + [key,]
                    # yield {'depth': depth-1, 'key': key, 'val': val, 'keylist': kl}
                    yield [depth-1, key, val, kl]
                # self.data_iterator(level=level, maxlevel=maxlevel, data=val)

                for d in self.data_iterator(
                        maxdepth=maxdepth, data=val, depth=depth, keylist=kl,
                        yield_short_kl=yield_short_kl):
                    yield d

        except:
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


if __name__ == '__main__':

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