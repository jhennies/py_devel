
import os
# from hdf5_image_processing import Hdf5ImageProcessingLib as IPL
from hdf5_slim_processing import RecursiveDict as Rdict
from hdf5_slim_processing import Hdf5Processing as Hp
import numpy as np
import processing_lib as lib
from yaml_parameters import YamlParams
from concurrent import futures
from timeit import default_timer as timer
from find_false_merges_src import FeatureImages, FeatureImageParams
import h5py
import re


__author__ = 'jhennies'


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


def compute_feature_images_multi(experiment, yparams):

    all_params = yparams.get_params()

    source = experiment['source']
    target = experiment['target']
    params = experiment['params']
    features = experiment['features']

    # Set source file
    source_file = all_params[source[0]] + all_params[source[1]]
    # Set target file
    target_file = all_params[target[0]] + all_params[target[1]]

    # -----------------------------------------------------------------
    # This gets me the internal paths that are desired according to the
    # regular expression input
    internal_paths = []

    search_pattern = source[2]['re']
    f = h5py.File(source_file, 'r')

    def recursive_visit(f, name=''):
        # print name
        if name != '':
            name += '/'
            if re.search(search_pattern, name):
                internal_paths.append(name)
        for k, v in f.iteritems():
            if type(v) is not h5py.Dataset:
                recursive_visit(v, name=name + k)
            else:
                if re.search(search_pattern, name + k):
                    # print name + k
                    internal_paths.append(name + k + '/')

    recursive_visit(f)
    # -----------------------------------------------------------------

    yparams.logging('Found internal paths: {}', internal_paths)

    feature_image_params = FeatureImageParams(
        feature_list=features,
        anisotropy=params['anisotropy'],
        max_threads_features=params['max_threads_features']
    )

    # Create a list of feature images objects, each referring to one internal path
    feature_images_list = [
        FeatureImages(
            source_filepath=all_params[source[0]] + all_params[source[1]],
            source_internal_path=x,
            filepath=target_file,
            internal_path=x,
            params=feature_image_params
        )
        for x in internal_paths
    ]

    # Actually compute the feature images
    yparams.logging('Starting thread pool with a max of {} threads', params['max_threads_sources'])
    with futures.ThreadPoolExecutor(params['max_threads_sources']) as do_stuff:
        # tasks = Hp()
        for feature_images in feature_images_list:
            do_stuff.submit(
                feature_images.compute_children
            )


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
