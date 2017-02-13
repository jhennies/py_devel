
import os
# from hdf5_image_processing import Hdf5ImageProcessingLib as IPL
from hdf5_slim_processing import RecursiveDict as Rdict
from hdf5_slim_processing import Hdf5Processing as Hp
import numpy as np
import processing_lib as lib
from yaml_parameters import YamlParams
from concurrent import futures
from timeit import default_timer as timer


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
        print 'Gaussian: Start...'
        gaussian = lib.gaussian_smoothing(image, specific_params[0], general_params['anisotropy'])
        print 'Gaussian: End!'
        return gaussian

    @staticmethod
    def disttransf(image, general_params, specific_params):
        print 'Distance transform: Start...'

        anisotropy = np.array(general_params['anisotropy']).astype(np.float32)
        image = image.astype(np.float32)

        # Compute boundaries
        axes = (anisotropy ** -1).astype(np.uint8)
        image = lib.pixels_at_boundary(image, axes)

        # Compute distance transform
        image = image.astype(np.float32)
        image = lib.distance_transform(image, pixel_pitch=anisotropy, background=True)

        print 'Distance transform: End!'
        return image

    @staticmethod
    def hessian_eigenvalues(image, general_params, specific_params):
        print 'Hessian: Start...'
        hessian = lib.hessian_of_gaussian_eigenvalues(
            image, specific_params[0], anisotropy=general_params['anisotropy']
        )
        print 'Hessian: End!'
        return hessian

    @staticmethod
    def structure_tensor_eigenvalues(image, general_params, specific_params):
        print 'Structure tensor: Start...'
        structen = lib.structure_tensor_eigenvalues(
            image, specific_params[0], specific_params[1],
            anisotropy=general_params['anisotropy']
        )
        print 'Structure tensor: End!'
        return structen

    @staticmethod
    def gaussian_gradient_magnitude(image, general_params, specific_params):
        print 'Gradient magnitude: Start...'
        mag = lib.gaussian_gradient_magnitude(
            image, specific_params[0],
            anisotropy=general_params['anisotropy']
        )
        print 'Gradient magnitude: End!'
        return mag

    @staticmethod
    def laplacian_of_gaussian(image, general_params, specific_params):
        print 'Laplacian: Start...'
        lapl = lib.laplacian_of_gaussian(
            image, specific_params[0],
            anisotropy=general_params['anisotropy']
        )
        print 'Laplacian: End!'
        return lapl


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

        function(final, yparams)


def return_img(image):
    return image


def compute_features_multi_wrapper(image, general_params, features, kl, target_file, logger=None):

    if kl is not None and logger is not None:
        logger.logging('Computing features on image {}', kl)

    result = Hp()

    result[kl] = compute_features_multi(image, general_params, features, logger=logger)
    image = None

    # Write result
    logger.logging('Writing in {}', target_file)
    result.write(target_file)


def compute_features_multi(image, general_params, features, logger=None):

    ff = FeatureFunctions()
    result = Hp()

    if type(image) is not np.ndarray:
        image = np.array(image)

    if logger is not None:
        logger.logging(
            'Starting thread pool with a max of {} threads', general_params['max_threads_features']
        )
    with futures.ThreadPoolExecutor(general_params['max_threads_features']) as do_stuff:

        keys = []
        vals = []
        tasks = Rdict()

        for k, v in features.iteritems():

            if v:

                tasks[k] = do_stuff.submit(getattr(ff, v.pop('func')), image, general_params, v.pop('params'))
                # print 'In the for loop: {}'.format(do_stuff.done())
                keys.append(k)
                vals.append(v)

            else:
                result[k] = image

    for k, v in dict(zip(keys, vals)).iteritems():
        result[k] = tasks[k].result()
        if len(v) > 0:
            result[k] = compute_features_multi(result[k], general_params, v)

    return result


def compute_features(image, general_params, features, logger=None, name=None):

    ff = FeatureFunctions()
    result = Hp()

    if type(image) is not np.ndarray:
        image = np.array(image)

    if name is not None and logger is not None:
        logger.logging('Computing features on image {}', name)

    for k, v in features.iteritems():

        if v:
            # print 'Computing {}'.format(v['func'])
            result[k] = getattr(ff, v.pop('func'))(image, general_params, v.pop('params'))

            if len(v) > 0:
                result[k] = compute_features(result[k], general_params, v)
        else:
            result[k] = image

    return result


def compute_feature_images_multi(experiment, yparams):

    all_params = yparams.get_params()

    source = experiment['source']
    target = experiment['target']
    params = experiment['params']
    features = experiment['features']

    # Load data
    if len(source) > 2:
        kw_load = source[2]
    data = load_images(
        all_params[source[0]] + all_params[source[1]], logger=yparams, **kw_load
    )

    yparams.logging('\nInitial datastructure: \n\n{}', data.datastructure2string(maxdepth=3))

    # Set target file
    target_file = all_params[target[0]] + all_params[target[1]]


    yparams.logging('Starting thread pool with a max of {} threads', params['max_threads_sources'])
    with futures.ThreadPoolExecutor(params['max_threads_sources']) as do_stuff:
        tasks = Hp()
        for d, k, v, kl in data.data_iterator(leaves_only=True):
            tasks[kl] = do_stuff.submit(
                compute_features_multi_wrapper, data[kl], params, features.dcp(),
                kl, target_file, logger=yparams
            )

        # # result = [x.result() for x in tasks]
        # for d, k, v, kl in tasks.data_iterator(leaves_only=True):
        #     data[kl] = v.result()


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
        all_params[source[0]] + all_params[source[1]], logger=yparams, **kw_load
    )

    yparams.logging('\nInitial datastructure: \n\n{}', data.datastructure2string(maxdepth=3))

    # Set target file
    target_file = all_params[target[0]] + all_params[target[1]]

    for d, k, v, kl in data.data_iterator(leaves_only=True):
        yparams.logging('===============================\nWorking on image: {}', kl)

        # Load data into memory
        data[kl] = np.array(v)

        # # Feature calculation
        # start = timer()
        # data[kl] = compute_features_multi(data[kl], params, features.dcp(), logger=yparams)
        # # data[kl] = None
        # elapsed = timer()
        # elapsed = elapsed - start
        # print "Time spent in compute_features_multi is: ", elapsed

        start = timer()
        data[kl] = compute_features(data[kl], params, features.dcp(), logger=yparams, name=kl)
        elapsed = timer()
        elapsed = elapsed - start
        print "Time spent in compute_features is: ", elapsed

        # # Write result
        # data.write(target_file, keys=kl)
        # Free memory
        data[kl] = None

    # Close file
    data.close()


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

        # experiment_parser(yparams, compute_feature_images, 'compute_feature_images')
        experiment_parser(yparams, compute_feature_images_multi, 'compute_feature_images')
        # compute_feature_images(yparams)

        yparams.logging('')
        yparams.stoplogger()

    except:
        raise
        yparams.errout('Unexpected error')


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters_ref.yml'

    run_compute_feature_images(yamlfile)
