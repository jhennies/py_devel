
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


__author__ = 'jhennies'


def load_images(ipl, ids):

    params = ipl.get_params()

    if 'rawdata' in ids:
        ipl.logging('Loading raw data from:\n{} ...', params['datafolder'] + params['rawdatafile'])
        ipl.data_from_file(
            params['datafolder'] + params['rawdatafile'],
            skeys=params['rawdataname'],
            recursive_search=True, nodata=True,
        )

    if 'probs' in ids:
        ipl.logging('Loading probabilities from:\n{} ...', params['datafolder'] + params['probsfile'])
        ipl.data_from_file(
            params['datafolder'] + params['probsfile'],
            skeys=params['probsname'],
            recursive_search=True, nodata=True
        )

    if 'largeobj' in ids:
        ipl.logging('Loading large objects from:\n{} ...', params['intermedfolder'] + params['largeobjfile'])
        ipl.data_from_file(
            params['intermedfolder'] + params['largeobjfile'],
            skeys=params['largeobjname'],
            recursive_search=True, nodata=True
        )

    if 'largeobjm' in ids:
        ipl.logging('Loading merged objects from:\n{} ...', params['intermedfolder'] + params['largeobjmfile'])
        ipl.data_from_file(
            params['intermedfolder'] + params['largeobjmfile'],
            skeys=params['largeobjmnames'][0],
            recursive_search=True, nodata=True
        )


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


def compute_features(image, general_params, subfeature_params):

    ff = FeatureFunctions()
    result = IPL()

    for k, v in subfeature_params.iteritems():

        if v:
            result[k] = getattr(ff, v.pop('func'))(image, general_params, v.pop('params'))

            if len(v) > 0:
                result[k] = compute_features(result[k], general_params, v)
        else:
            result[k] = image

    return result

def compute_selected_features(ipl, params):

    thisparams = rdict(data=params['compute_feature_images'])
    targetfile = params['intermedfolder'] + params['featureimsfile']

    maxd = ipl.maxdepth()

    for d, k, v, kl in  ipl.data_iterator(yield_short_kl=True):

        if d == maxd:
            ipl.logging('----------------------\nWorking on image: {}', k)

            ipl[kl].populate(k)

            if k in [params['rawdataname'], params['probsname'], params['largeobjname'], params['largeobjmnames'][0]]:
                general_params = thisparams.dcp()
                del general_params['features']

                if k == params['rawdataname']:
                    subfeature_params = thisparams['features']['rawdata']
                    ipl[kl][k] = compute_features(ipl[kl][k], general_params, subfeature_params)
                elif k == params['probsname']:
                    subfeature_params = thisparams['features']['probs']
                    ipl[kl][k] = compute_features(ipl[kl][k], general_params, subfeature_params)
                elif k == params['largeobjname']:
                    subfeature_params = thisparams['features']['largeobj']
                    ipl[kl][k] = compute_features(ipl[kl][k], general_params, subfeature_params)
                elif k == params['largeobjmnames'][0]:
                    subfeature_params = thisparams['features']['largeobjm']
                    ipl[kl][k] = compute_features(ipl[kl][k], general_params, subfeature_params)

                ipl.write(filepath=targetfile, keys=[kl + [k]])
                ipl[kl][k] = None


def compute_feature_images(ipl):

    params = ipl.get_params()
    thisparams = rdict(params['compute_feature_images'])
    # targetfile = params['intermedfolder'] + params['featureimsfile']

    # Load the necessary images
    load_images(ipl, thisparams['features'].keys())

    ipl.logging('\nInitial datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

    maxd = ipl.maxdepth()
    for d, k, v, kl in ipl.data_iterator(maxdepth=ipl.maxdepth() - 1):

        if d == maxd - 1:

            ipl.logging('===============================\nWorking on group: {}', kl)

            # TODO: Implement copy full logger
            ipl[kl].set_logger(ipl.get_logger())

            # # Load the image data into memory
            # ipl[kl].populate(k)

            compute_selected_features(ipl.subset(kl), params)

            # Write the result to file
            # ipl.write(filepath=targetfile, keys=[kl + [k]])
            # # Free memory (With this command the original reference to the source file is restored)
            # ipl[kl].unpopulate()


def run_compute_feature_images(yamlfile):

    ipl = IPL(yaml=yamlfile)

    ipl.set_indent(1)

    params = rdict(data=ipl.get_params())
    ipl.startlogger(filename=params['resultfolder'] + 'compute_feature_images.log', type='w', name='ComputeFeatureImages')

    try:

        # # Copy the script file and the parameters to the scriptsfolder
        # copy(inspect.stack()[0][1], params['scriptsfolder'])
        # copy(yamlfile, params['scriptsfolder'] + 'compute_feature_images.parameters.yml')

        # ipl.logging('\nInitial datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

        compute_feature_images(ipl)

        # ipl.logging('\nFinal datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

        # ipl.write(filepath=params['intermedfolder'] + params['largeobjfile'])

        ipl.logging('')
        ipl.stoplogger()

    except:

        ipl.errout('Unexpected error')


if __name__ == '__main__':
    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'

    run_compute_feature_images(yamlfile)
