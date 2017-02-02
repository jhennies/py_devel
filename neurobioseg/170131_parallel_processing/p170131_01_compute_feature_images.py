
import os
# from hdf5_image_processing import Hdf5ImageProcessingLib as IPL
from hdf5_slim_processing import RecursiveDict as Rdict
from hdf5_slim_processing import Hdf5Processing as Hp
import numpy as np
import processing_lib as lib
from yaml_parameters import YamlParams


__author__ = 'jhennies'


def load_images(filepath, skeys=None, recursive_search=False, logger=None):

    if logger is not None:
        logger.logging('Loading data from \n{}', filepath)
    else:
        print 'Loading data from \n{}'.format(filepath)

    data = Hp()

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


def experiment_parser(yparams, function, name):

    all_params = yparams.get_params()

    # Zero'th layer:
    # --------------
    zeroth = Rdict(all_params[name])
    if 'default' in zeroth:
        zeroth_defaults = zeroth.pop('default')
    else:
        zeroth_defaults = Rdict()

    for exp_lbl, experiment in zeroth.iteritems():

        # First layer
        # -----------
        # An experiment is now selected and performed
        yparams.logging('\n\nPerforming experiment {}\n==============================', exp_lbl)

        final = zeroth_defaults.dcp()
        final.merge(experiment)

        compute_feature_images(final, yparams)


def compute_feature_images(experiment, yparams):

    all_params = yparams.get_params()

    source = experiment['source']
    target = experiment['target']
    params = experiment['params']
    features = experiment['features']

    # Load data
    if len(source) > 2:
        kw_load = source[2]
    data = load_images(
        all_params[source[0]] + params[source[1]], logger=yparams, **kw_load
    )

    yparams.logging('\nInitial datastructure: \n\n{}', data.datastructure2string(maxdepth=3))

    # Set target file
    target_file = all_params[target[0]] + all_params[target[1]]



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

        experiment_parser(yparams, compute_feature_images, 'compute_feature_images')
        # compute_feature_images(yparams)

        yparams.logging('')
        yparams.stoplogger()

    except:

        yparams.errout('Unexpected error')


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters_ref.yml'

    run_compute_feature_images(yamlfile)
