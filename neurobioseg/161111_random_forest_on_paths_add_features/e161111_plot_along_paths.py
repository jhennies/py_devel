
from hdf5_image_processing import Hdf5ImageProcessing as ip, Hdf5ImageProcessingLib as ipl
from hdf5_processing import RecursiveDict as rdict, Hdf5Processing as hp
import numpy as np
import os
import traceback
import matplotlib.pyplot as plt
import matplotlib.pylab as lab
import processing_lib as lib
import sys

# Done: Visualize Distancetransform along paths of true and false merges
# Done: Also intensity values of raw data
# TODO: And probability map
# Done: Do not forget to save the result for the thesis!


__author__ = 'jhennies'


def load_images(dict_main):
    """
    These images are loaded:
    paths_true (paths within single label objects)
    paths_false (paths of merged objects which cross the merging site)
    featureims_true
    featureims_false
    :param dict_main:
    :return:
    """
    paths_true = ipl()
    paths_false = ipl()
    featureims_true = ipl()
    featureims_false = ipl()

    params = dict_main.get_params()

    dict_main.logging('Loading true paths ...')
    # Paths within labels (true paths)
    paths_true.data_from_file(
        filepath=params['intermedfolder'] + params['pathstruefile'],
        skeys=[['x', '0', 'path']],
        recursive_search=False, nodata=True
    )

    dict_main.logging('Loading false paths ...')
    # Paths of merges (false paths)
    paths_false.data_from_file(
        filepath=params['intermedfolder'] + params['pathsfalsefile'],
        skeys=[['x', '0', 'path']],
        recursive_search=False, nodata=True
    )

    dict_main.logging('Loading features for true paths ...')
    # Load features for true paths
    featureims_true.data_from_file(
        filepath=params['intermedfolder'] + params['featureimsfile'],
        nodata=True, skeys=[['x', '0']]
    )
    featureims_true.delete_items(params['largeobjmnames'][0])

    dict_main.logging('Loading features for false paths ...')
    # Load features for false paths
    featureims_false.data_from_file(
        filepath=params['intermedfolder'] + params['featureimsfile'],
        nodata=True, skeys=[['x', '0']]
    )
    featureims_false.delete_items(params['largeobjname'])

    return (paths_true, paths_false, featureims_true, featureims_false)


def kl2str(kl):
    klstr = ''
    for i in kl:
        klstr += i
    return klstr


def plot_paths(paths, featureims, folder, name):

    for d, k, v, kl in paths.data_iterator():
        if type(v) is not type(paths):

            pathvals = []
            print '{}'.format(k)
            for df, kf, vf, klf in featureims.data_iterator():
                if type(vf) is not type(featureims):

                    print '    {}'.format(kf)
                    pathvals.append(lib.getvaluesfromcoords(vf, v))

            x = range(0, len(pathvals[0]))
            print np.array(pathvals).shape
            pathvals = np.swapaxes(pathvals, 0, 1)
            print x

            plt.plot(x, pathvals)
            # plt.show()
            print folder + 'plots/' + name + kl2str(kl) + '.png'
            print kl2str(kl)
            print kl
            lab.savefig(folder + 'plots/' + name + kl2str(kl) + '.png')
            plt.clf()


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters_ref.yml'
    dict_main = ipl(yaml=yamlfile)
    
    paths_true, paths_false, featureims_true, featureims_false = load_images(dict_main)

    dict_main.startlogger()

    dict_main.logging('\npaths_true datastructure\n---\n{}', paths_true.datastructure2string(maxdepth=2))
    dict_main.logging('\npaths_false datastructure\n---\n{}', paths_false.datastructure2string(maxdepth=2))
    dict_main.logging('\nfeatureims_true datastructure\n---\n{}', featureims_true.datastructure2string(maxdepth=2))
    dict_main.logging('\nfeatureims_false datastructure\n---\n{}', featureims_false.datastructure2string(maxdepth=2))

    params = dict_main.get_params()
    plot_paths(paths_true, featureims_true, params['intermedfolder'], '_true_')
    plot_paths(paths_false, featureims_false, params['intermedfolder'], '_false_')

    sys.exit()
    try:

        # hfp.logging('hfp datastructure:\n---\n{}---', hfp.datastructure2string(maxdepth=2))

        # # Pop selected labels
        # a = hfp['true', 'border']
        # a.pop('63')

        for k, v in a.iteritems():
            print k
            for k2, v2 in v.iteritems():
                print k2
                hfp.anytask(lib.getvaluesfromcoords, v2,
                            reciprocal=False,
                            keys=('disttransf', 'raw'),
                            tkeys='{}.{}.{}'.format('result_true', k, k2))

                #'{}.{}.{}'.format('result_true', k, k2)


        a = hfp['false', 'border']
        # a.pop('63')

        for k, v in a.iteritems():
            print k
            for k2, v2 in v.iteritems():
                print k2
                hfp.anytask(lib.getvaluesfromcoords, v2,
                          reciprocal=False,
                          keys=('disttransfm', 'raw'),
                          tkeys='{}.{}.{}'.format('result_false', k, k2))

        hfp.pop('disttransf')
        hfp.pop('disttransfm')
        # hfp.pop('result_false')
        # hfp.pop('result_true')
        hfp.pop('true')
        hfp.pop('false')
        hfp.pop('raw')

        hfp.logging('hfp datastructure:\n---\n{}---', hfp.datastructure2string(maxdepth=2))

        # print hfp['result.6720_13067.8']

        y = []
        for k, v in hfp.iteritems():

            if v.values()[0]:

                try:
                    # y.append(v)
                    x = range(0, len(v.values()[0]))
                    y = np.swapaxes(np.array(v.values()), 0, 1)
                    plt.plot(x, y)

                    # plt.show()
                    lab.savefig(params['intermedfolder'] + 'plots/' + k + '.png')
                    plt.clf()
                except ValueError:
                    pass

    except:

        hfp.errout('Unexpected error', traceback)

    hfp.stoplogger()