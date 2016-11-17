
from p161111_00_remove_small_objects import run_remove_small_objects
from p161111_01_merge_adjacent_objects import run_merge_adjacent_objects
from p161111_02_compute_feature_images import run_compute_feature_images
from yaml_parameters import YamlParams
import os
from shutil import copy
import inspect
import argparse
from hdf5_processing import RecursiveDict as rd

__author__ = 'jhennies'


if __name__ == '__main__':

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Run post-processing pipeline for neuron segmentation')
    parser.add_argument('ResultFolder', type=str, help='The results and intermediate results are stored to this location')
    parser.add_argument('ParameterFile', type=str, help='The parameter file in yaml format')

    args = parser.parse_args()

    resultfolder = args.ResultFolder
    yamlfile = args.ParameterFile

    yaml = YamlParams(filename=yamlfile)
    params = yaml.get_params()

    yaml.startlogger(filename=params['resultfolder'] + 'pipeline.log', type='w', name='Pipeline')

    yaml.logging('Starting script with:')
    yaml.logging('    ResultFolder = {}', resultfolder)
    yaml.logging('    ParameterFile = {}\n', yamlfile)

    # Create folder for scripts
    if not os.path.exists(params['scriptsfolder']):
        os.makedirs(params['scriptsfolder'])
    else:
        if params['overwriteresults']:
            yaml.logging(
                'Warning: Scriptsfolder already exists and content will be overwritten...\n')
        else:
            raise IOError('Error: Scriptsfolder already exists!')

    # Create folder for intermediate results
    if not os.path.exists(params['intermedfolder']):
        os.makedirs(params['intermedfolder'])
    else:
        if params['overwriteresults']:
            yaml.logging(
                'Warning: Intermedfolder already exists and content will be overwritten...\n')
        else:
            raise IOError('Error: Intermedfolder already exists!')

    # Copy script and parameter file to the script folder
    copy(inspect.stack()[0][1], params['scriptsfolder'])
    copy(yamlfile, params['scriptsfolder'] + 'pipeline.parameters.yml')

    # The pipeline
    # _________________________________________________________________________________________
    if params['run_remove_small_objects']:
        yaml.logging('Removing small objects ...')
        run_remove_small_objects(yaml.get_filename())

    if params['run_merge_adjacent_objects']:
        yaml.logging('Merging adjacent objects ...')
        run_merge_adjacent_objects(yaml.get_filename())

    if params['run_compute_feature_images']:
        yaml.logging('Computing feature images ...')
        run_compute_feature_images(yaml.get_filename())
    # _________________________________________________________________________________________

    yaml.stoplogger()

