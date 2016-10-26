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

    def __init__(self, path=None, filename=None, filepath=None, data=None, skeys=None, tkeys=None, castkey=None,
                 yaml=None, yamlspec=None):

        dict.__init__(self)
        YamlParams.__init__(self, filename=yaml)

        if yaml is not None:

            yamldict = self.get_params()
            if yamlspec is None:
                if 'filepath' in yamldict.keys(): filepath = yamldict['filepath']
                if 'path' in yamldict.keys(): filepath = yamldict['path']
                if 'filename' in yamldict.keys(): filepath = yamldict['filename']
                if 'castkey' in yamldict.keys(): castkey = yamldict['castkey']
                if 'skeys' in yamlspec.keys():
                    if type(yamlspec['skeys']) is dict:
                        skeys = ()
                        for i in xrange(1, len(yamlspec['skeys'])):
                            skeys += (yamldict[yamlspec['skeys'][0]][yamlspec['skeys'][i]],)
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
            self.data_from_file(path+filename, skeys=skeys, tkeys=tkeys, castkey=castkey)

        elif filepath is not None:
            self.data_from_file(filepath, skeys=skeys, tkeys=tkeys, castkey=castkey)

    def __getitem__(self, item):

        # # This atomatically creates items that are not there in the first place
        # try:
        #     return dict.__getitem__(self, item)
        # except KeyError:
        #     value = self[item] = type(self)()
        #     return value

        # if type(item) is tuple:
        #     execstr = ('self' + "[{}]" * len(item)).format(*item)
        #     print execstr

        if type(item) is tuple or type(item) is list:

            item = list(item)
            if len(item) > 1:
                firstkey = item.pop(0)
                # try:
                # return Hdf5Processing(data=self[firstkey])[item]
                return type(self)(data=self[firstkey])[item]
                # except KeyError:
                #     value = Hdf5Processing(data=self[firstkey])[item] = type(self)()
                #     return value
            else:
                return dict.__getitem__(self, item[0])

            # except KeyError:
            #     value = self[item.pop] = type(self)()
            #     return value

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
                # super(Hdf5Processing, self).__setitem__(key[0], val)
                dict.__setitem__(self, key[0], val)
        else:
            # super(Hdf5Processing, self).__setitem__(key, val)
            dict.__setitem__(self, key, val)

    def setdata(self, data, tkeys=None):

        if tkeys is not None:
            if type(tkeys) is not tuple and type(tkeys) is not list:
                tkeys = (tkeys,)

            if len(tkeys) != len(data.keys()):
                if len(tkeys) == 1:
                    # tkeys = [tkeys[0] + x for x in data.keys()]
                    tkeys = zip(tkeys * len(data.keys()), data.keys())
                else:
                    raise RuntimeError('Hdf5Processing: Length of tkeys must be equal to length of data keys!')

        else:
            tkeys = data.keys()

        for i in xrange(0, len(data.keys())):
            try:
                self[tkeys[i]] = type(self)(data=data[data.keys()[i]])
            except:
                self[tkeys[i]] = data[data.keys()[i]]

        # for key, val in data.iteritems():
        #     try:
        #         self[key] = type(self)(data=val)
        #     except AttributeError:
        #         self[key] = val

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

                    group.create_dataset(k, data=v)

                else:
                    print 'Warning in Hdf5Processing.write(): Nothing to write.'

    def write(self, filepath=None, of=None):

        if of is None:
            of = h5py.File(filepath)

        self.write_dataset(of, self)

        of.close()

    def get_h5_content(self, f, skeys=None, offset='    ', castkey=None):

        # if isinstance(f, h5py.Dataset):
        #     print offset, '(Dataset)', f.name, 'len =', f.shape
        #
        # if isinstance(f, h5py.Group):
        #     print offset, '(Group)', f.name
        #
        # else :
        #     pass
        #     # print 'WORNING: UNKNOWN ITEM IN HDF5 FILE', f.name
        #     # sys.exit ( "EXECUTION IS TERMINATED" )
        # dict_f = {}

        if isinstance(f, h5py.File) or isinstance(f, h5py.Group):
            dict_f = dict(f)
            if skeys is None:
                skeys = dict_f.keys()
            rtrn_dict = {}
            for key, val in dict_f.iteritems():
                subg = val
                # print offset, key
                if castkey is not None:
                    key = castkey(key)
                    # print '{} type(key) = {}'.format(offset, type(key))
                if key in skeys:
                    rtrn_dict[key] = self.get_h5_content(subg, offset=offset + '    ', castkey=castkey)

        else:
            # print offset, f
            rtrn_dict = np.array(f)

        return rtrn_dict

    def load_h5(self, filepath, skeys=None, castkey=None):

        f = h5py.File(filepath)

        return self.get_h5_content(f, skeys=skeys, castkey=castkey)

    def data_from_file(self, filepath, skeys=None, tkeys=None, castkey=None):
        newdata = self.load_h5(filepath, skeys=skeys, castkey=castkey)
        self.setdata(newdata, tkeys=tkeys)

    def getdataitem(self, itemkey):
        return self[itemkey]

    def datastructure2string(self, data=None, dstr='', indent=0, maxdepth=None, depth=0, indentstr='.  '):

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

    def data_iterator(self, maxdepth=None, data=None, depth=0, keylist=[]):

        depth += 1
        if maxdepth is not None:
            if depth-1 > maxdepth:
                return

        if data is None:
            data = self

        try:

            for key, val in data.iteritems():
                # print key, val
                kl = keylist + [key,]
                # yield {'depth': depth-1, 'key': key, 'val': val, 'keylist': kl}
                yield [depth-1, key, val, kl]
                # self.data_iterator(level=level, maxlevel=maxlevel, data=val)

                for d in self.data_iterator(maxdepth=maxdepth, data=val, depth=depth, keylist=kl):
                    yield d

        except:
            pass

    # # ANYTASK-based functions
    #
    # def anytask(self, task, *args, **kwargs):
    #     """
    #     :param task:
    #
    #     :param args:
    #
    #     :param kwargs:
    #
    #     """
    #
    #     for d in self.data_iterator(maxdepth=None):
    #
    #         if type(d['val']) is not dict:
    #
    #             d['val'] = task(d['val'], *args, **kwargs)
    #             self._data[d['keylist']] = d['val']


if __name__ == '__main__':

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


    hfp = Hdf5Processing(data={'a': {'b': np.array([10, 10])}})

    hfp['a', 'c'] = np.array([20, 10, 20, 30])
    hfp['d', 'b'] = [[1, 1, 2], [2, 3, 4], [3, 2, 1], [2], [12, 3], [4, 3,4]]

    print hfp
    print hfp.datastructure2string()

    hfp.write(filepath='/home/julian/Documents/test.h5')