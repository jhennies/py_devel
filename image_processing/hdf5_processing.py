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


class Hdf5Processing(YamlParams):

    _data = None

    def __init__(self, path=None, filename=None, filepath=None, data=None, dataname=None, castkey=None,
                 yaml=None, yamlspec=None):

        YamlParams.__init__(self, filename=yaml)

        if yaml is not None:

            yamldict = self.get_params()
            if yamlspec is None:
                if 'filepath' in yamldict.keys(): filepath = yamldict['filepath']
                if 'path' in yamldict.keys(): filepath = yamldict['path']
                if 'filename' in yamldict.keys(): filepath = yamldict['filename']
                if 'dataname' in yamldict.keys(): dataname = yamldict['dataname']
                if 'castkey' in yamldict.keys(): castkey = yamldict['castkey']

            else:
                if 'filepath' in yamlspec.keys(): filepath = yamldict[yamlspec['filepath']]
                if 'path' in yamlspec.keys(): path = yamldict[yamlspec['path']]
                if 'filename' in yamlspec.keys(): filename = yamldict[yamlspec['filename']]
                if 'dataname' in yamlspec.keys(): dataname = yamldict[yamlspec['dataname']]
                if 'castkey' in yamlspec.keys(): castkey = yamldict[yamlspec['castkey']]


        if data is not None:
            self.setdata(data, dataname)

        elif path is not None:
            self.data_from_file(path+filename, dataname, castkey=castkey, append=False)

        elif filepath is not None:
            self.data_from_file(filepath, dataname, castkey=castkey, append=False)

    def setdata(self, data, append=True):

        if not append or self._data is None:
            self._data = {}

        for d in data:
            self._data[d] = data[d]

    def write(self, filepath):

        of = h5py.File(filepath)

        for dk, dv in self._data.items():

            if type(dv) is list:

                grp = of.create_group(str(dk))
                for i in xrange(0, len(dv)):
                    grp.create_dataset(str(i), data=dv[i])

            elif type(dv) is np.array:

                of.create_dataset(dk, data=dv)

            else:
                print 'Warning in Hdf5Processing.write(): Nothing to write.'

        of.close()

    def get_h5_content(self, f, offset='    ', castkey=None):

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
            rtrn_dict = {}
            for key, val in dict_f.iteritems():
                subg = val
                # print offset, key
                if castkey is not None:
                    key = castkey(key)
                    # print '{} type(key) = {}'.format(offset, type(key))
                rtrn_dict[key] = self.get_h5_content(subg, offset + '    ', castkey=castkey)

        else:
            # print offset, f
            rtrn_dict = np.array(f)

        return rtrn_dict

    def load_h5(self, filepath, castkey=None):

        f = h5py.File(filepath)

        return self.get_h5_content(f, castkey=castkey)

    def data_from_file(self, filepath, dataname, castkey=None, append=True):
        newdata = self.load_h5(filepath, castkey=castkey)
        self.setdata({dataname: newdata}, append)

    def getdata(self):
        return self._data

    def getdataitem(self, itemkey):
        return self._data[itemkey]

    def keys(self):
        return self._data.keys()

    def datastructure2string(self, data=None, dstr='', indent=0, maxdepth=None, depth=0):

        depth += 1
        if maxdepth is not None:
            if depth > maxdepth:
                return dstr

        if data is None:
            data = self._data

        if type(data) is dict:

            for key, val in data.iteritems():

                # print key
                dstr += '{}{}\n'.format(' '*indent, key)
                dstr = self.datastructure2string(data=val, dstr=dstr, indent=indent+4, maxdepth=maxdepth, depth=depth)

        return dstr

    def data_iterator(self, maxdepth=None, data=None, depth=0, keylist=[]):

        depth += 1
        if maxdepth is not None:
            if depth-1 > maxdepth:
                return

        if data is None:
            data = self._data

        if type(data) is dict:

            for key, val in data.iteritems():
                # print key, val
                kl = keylist + [key,]
                yield {'depth': depth-1, 'key': key, 'val': val, 'keylist': kl}
                # self.data_iterator(level=level, maxlevel=maxlevel, data=val)

                for d in self.data_iterator(maxdepth=maxdepth, data=val, depth=depth, keylist=kl):
                    yield d

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

    hfp = Hdf5Processing(
        filepath='/media/julian/Daten/neuraldata/cremi_2016/develop/161011_locmax_paths_feature_extraction/intermediate/cremi.splA.raw_neurons.crop.crop_10-200-200_110-712-712.paths.true.h5',
        dataname='content',
        castkey=float
    )

    data = hfp.getdataitem('content')
    print data.keys()
    print data[data.keys()[0]].keys()
    print data[data.keys()[0]][data[data.keys()[0]].keys()[0]]