
from hdf5_image_processing import Hdf5ImageProcessing as IP, Hdf5ImageProcessingLib as IPL
from hdf5_processing import Hdf5Processing as HP
import numpy as np
import os
import traceback
import matplotlib.pyplot as plt
import matplotlib.pylab as lab
import processing_lib as lib

# TODO: Visualize Distancetransform along paths of true and false merges
# TODO: Also intensity values or raw data and probability map
# TODO: Do not forget to save the result for the thesis!


__author__ = 'jhennies'


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'
    hfp = IPL(
        yaml=yamlfile,
        yamlspec={'path': 'intermedfolder', 'filename': 'pathstruefile'},
        tkeys='true',
        castkey=None
    )

    params = hfp.get_params()

    hfp.data_from_file(
        filepath=params['intermedfolder'] + params['pathsfalsefile'],
        tkeys='false',
        castkey=None
    )

    hfp.data_from_file(
        filepath=params['intermedfolder'] + params['locmaxfile'],
        skeys=('disttransf', 'disttransfm'),
        tkeys=('disttransf', 'disttransfm')
    )

    hfp.startlogger()

    try:

        hfp.logging('hfp datastructure:\n---\n{}---', hfp.datastructure2string(maxdepth=2))

        hfp.anytask(lib.getvaluesfromcoords,
            reciprocal=True,
            keys='disttransfm',
            indict=hfp['false', '6155_9552'],
            tkeys='result_false'
        )

        hfp.logging('hfp datastructure:\n---\n{}---', hfp.datastructure2string(maxdepth=2))

        # y = []
        # maxlen = 0
        # for d, k, v, kl in hfp['result_false'].data_iterator():
        #     y.append(v)
        #     x = range(0, len(v))
        #     plt.plot(x, v)
        #     if len(v) > maxlen:
        #         maxlen = len(v)
        # # x = range(0, len(hfp['path_dt_6155_9552_0']))
        # # y = hfp['path_dt_6155_9552_0']
        # hfp.logging('y = {}', y)
        # hfp.logging('len(y) = {}', len(y))
        # x = range(0, maxlen)
        #
        # plt.show()

        for k, v in hfp['false'].iteritems():
            print k
            for k2, v2 in v.iteritems():
                print k2
                hfp.anytask(lib.getvaluesfromcoords, v2,
                          reciprocal=False,
                          keys='disttransfm',
                          tkeys='{}.{}.{}'.format('result_false', k, k2))

        # for k, v in hfp['true'].iteritems():
        #     print k
        #     for k2, v2 in v.iteritems():
        #         print k2
        #         hfp.anytask(lib.getvaluesfromcoords, v2,
        #                   reciprocal=False,
        #                   keys='disttransf',
        #                   tkeys='{}.{}.{}'.format('result_true', k, k2))

        hfp.pop('disttransf')
        hfp.pop('disttransfm')
        hfp.pop('result_false')
        # hfp.pop('result_true')
        hfp.pop('true')
        hfp.pop('false')

        hfp.logging('hfp datastructure:\n---\n{}---', hfp.datastructure2string(maxdepth=2))

        print hfp['result.6720_13067.8']

        y = []
        for k, v in hfp.iteritems():

            if v:

                try:
                    y.append(v)
                    x = range(0, len(v))
                    plt.plot(x, v)

                    # plt.show()
                    lab.savefig(params['intermedfolder'] + 'plots/' + k + '.png')
                    plt.clf()
                except ValueError:
                    pass

    except:

        hfp.errout('Unexpected error', traceback)

    hfp.stoplogger()