
import os
import inspect
from hdf5_image_processing import Hdf5ImageProcessing as IP, Hdf5ImageProcessingLib as IPL
from hdf5_processing import RecursiveDict as rdict
from shutil import copy, copyfile
import numpy as np
import matplotlib.pyplot as plt
import processing_libip as libip
import sys
import processing_lib as lib
from yaml_parameters import YamlParams


__author__ = 'jhennies'


def load_images(filepath, skeys=None, recursive_search=False, logger=None):

    if logger is not None:
        logger.logging('Loading data from \n{}', filepath)
    else:
        print 'Loading data from \n{}'.format(filepath)

    data = IPL()

    data.data_from_file(
        filepath=filepath,
        skeys=skeys,
        recursive_search=recursive_search,
        nodata=True
    )

    return data


class FeatureFunctions:

    def __init__(self):
        pass

    @staticmethod
    def gaussian(image, general_params, specific_params):
        return lib.gaussian_smoothing(image, specific_params[0], general_params['anisotropy'])

    @staticmethod
    def disttransf(image, general_params, specific_params):

        anisotropy = np.array(general_params['anisotropy']).astype(np.float32)
        image = image.astype(np.float32)

        # Compute boundaries
        axes = (anisotropy ** -1).astype(np.uint8)
        image = lib.pixels_at_boundary(image, axes)

        # Compute distance transform
        image = image.astype(np.float32)
        image = lib.distance_transform(image, pixel_pitch=anisotropy, background=True)

        return image

    @staticmethod
    def hessian_eigenvalues(image, general_params, specific_params):
        return lib.hessian_of_gaussian_eigenvalues(
            image, specific_params[0], anisotropy=general_params['anisotropy']
        )

    @staticmethod
    def structure_tensor_eigenvalues(image, general_params, specific_params):
        return lib.structure_tensor_eigenvalues(
            image, specific_params[0], specific_params[1],
            anisotropy=general_params['anisotropy']
        )

    @staticmethod
    def gaussian_gradient_magnitude(image, general_params, specific_params):
        return lib.gaussian_gradient_magnitude(
            image, specific_params[0],
            anisotropy=general_params['anisotropy']
        )

    @staticmethod
    def laplacian_of_gaussian(image, general_params, specific_params):
        return lib.laplacian_of_gaussian(
            image, specific_params[0],
            anisotropy=general_params['anisotropy']
        )


def compute_features(image, general_params, subfeature_params):

    ff = FeatureFunctions()
    result = IPL()

    for k, v in subfeature_params.iteritems():

        if v:
            print 'Computing {}'.format(v['func'])
            result[k] = getattr(ff, v.pop('func'))(image, general_params, v.pop('params'))

            if len(v) > 0:
                result[k] = compute_features(result[k], general_params, v)
        else:
            result[k] = image

    return result


# def compute_selected_features(ipl, params):
#
#     thisparams = rdict(data=params['compute_feature_images'])
#     targetfile = params['intermedfolder'] + params['featureimsfile']
#
#     maxd = ipl.maxdepth()
#
#     for d, k, v, kl in ipl.data_iterator(yield_short_kl=True):
#
#         if d == maxd:
#             ipl.logging('----------------------\nWorking on image: {}', k)
#
#             ipl[kl].populate(k)
#
#             if k in [params['rawdataname'], params['probsname'], params['largeobjname'], params['largeobjmnames'][0]]:
#                 general_params = thisparams.dcp()
#                 del general_params['features']
#
#                 if k == params['rawdataname']:
#                     subfeature_params = thisparams['features']['rawdata']
#                     ipl[kl][k] = compute_features(ipl[kl][k], general_params, subfeature_params)
#                 elif k == params['probsname']:
#                     subfeature_params = thisparams['features']['probs']
#                     ipl[kl][k] = compute_features(ipl[kl][k], general_params, subfeature_params)
#                 elif k == params['largeobjname']:
#                     subfeature_params = thisparams['features']['largeobj']
#                     ipl[kl][k] = compute_features(ipl[kl][k], general_params, subfeature_params)
#                 elif k == params['largeobjmnames'][0]:
#                     subfeature_params = thisparams['features']['largeobjm']
#                     ipl[kl][k] = compute_features(ipl[kl][k], general_params, subfeature_params)
#
#                 ipl.write(filepath=targetfile, keys=[kl + [k]])
#                 ipl[kl][k] = None


def compute_feature_images(yparams):

    params = yparams.get_params()
    thisparams = rdict(params['compute_feature_images'])
    # targetfile = params['intermedfolder'] + params['featureimsfile']
    general_params = thisparams['general_params']

    for sourcekey, source in thisparams['sources'].iteritems():

        # Load the necessary images
        #   1. Determine the settings for fetching the data
        try:
            recursive_search = False
            recursive_search = thisparams['skwargs', 'default', 'recursive_search']
            recursive_search = thisparams['skwargs', sourcekey, 'recursive_search']
        except KeyError:
            pass
        if len(source) > 2:
            skeys = source[2]
        else:
            skeys = None

        #   2. Load the data
        data = load_images(
            params[source[0]] + params[source[1]], skeys=skeys, recursive_search=recursive_search,
            logger=yparams
        )

        # Set targetfile
        targetfile = params[thisparams['targets', sourcekey][0]] \
                     + params[thisparams['targets', sourcekey][1]]

        yparams.logging('\nInitial datastructure: \n\n{}', data.datastructure2string(maxdepth=3))

        for d, k, v, kl in data.data_iterator(yield_short_kl=True, leaves_only=True):

            yparams.logging('===============================\nWorking on image: {}', kl + [k])

            # # TODO: Implement copy full logger
            # data[kl].set_logger(data.get_logger())

            # Load the image data into memory
            data[kl].populate(k)

            # compute_selected_features(data.subset(kl), params)
            subfeature_params = thisparams['features', sourcekey].dcp()
            data[kl][k] = compute_features(data[kl][k], general_params, subfeature_params)

            # Write the result to file
            data.write(filepath=targetfile, keys=[kl + [k]])
            # Free memory
            data[kl][k] = None


def run_compute_feature_images(yamlfile):

    yparams = YamlParams(filename=yamlfile)
    params = yparams.get_params()

    # Logger stuff
    yparams.set_indent(1)
    yparams.startlogger(
        filename=params['resultfolder'] + 'compute_feature_images.log',
        type='w', name='ComputeFeatureImages'
    )

    try:

        compute_feature_images(yparams)

        yparams.logging('')
        yparams.stoplogger()

    except:
        raise
        yparams.errout('Unexpected error')


if __name__ == '__main__':
    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters_ref.yml'

    run_compute_feature_images(yamlfile)
