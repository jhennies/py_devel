
import sys
sys.path.append('../../image_processing/')

from p161111_00_remove_small_objects import run_remove_small_objects
from p161111_01_merge_adjacent_objects import run_merge_adjacent_objects
from p161111_02_compute_feature_images import run_compute_feature_images
from p161111_03_find_border_contacts import run_find_border_contacts
from p161111_04a_paths_of_labels import run_paths_of_labels
from p161111_04b_paths_of_merges import run_paths_of_merges
from p161111_05_features_of_paths import run_features_of_paths
from p161111_06a_random_forest import run_random_forest
from yaml_parameters import YamlParams
import os
from shutil import copy
import inspect
import argparse
from hdf5_processing import RecursiveDict as rd

__author__ = 'jhennies'


from tempfile import mkstemp
from shutil import move
from os import remove, close
import re

def replace(file_path, pattern, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with open(abs_path,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                newline = re.sub(pattern, subst, line)
                new_file.write(newline)
                # new_file.write(line.replace(pattern, subst))
    close(fh)
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)


if __name__ == '__main__':

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Run post-processing pipeline for neuron segmentation')
    parser.add_argument(
        'ParameterFile', type=str,
        help='The parameter file in yaml format'
    )
    parser.add_argument(
        '-f', '--ResultFolder', type=str, nargs=1,
        help='Result folder, when specified result folder in parameter file will be ignored'
    )

    args = parser.parse_args()

    yamlfile = args.ParameterFile
    resultfolder = None
    if args.ResultFolder is not None:
        resultfolder = args.ResultFolder[0]

    if resultfolder is not None:
        replace('./parameters.yml', '^resultfolder: .*$', "resultfolder: {}".format(resultfolder))
        replace('./parameters.yml', '^intermedfolder: .*$', 'intermedfolder: {}{}'.format(resultfolder, 'intermed/'))
        replace('./parameters.yml', '^scriptsfolder: .*$', 'scriptsfolder: {}{}'.format(resultfolder, 'scripts/'))
        if not os.path.exists(resultfolder):
            os.makedirs(resultfolder)

    yaml = YamlParams(filename=yamlfile)
    params = yaml.get_params()

    yaml.startlogger(filename=params['resultfolder'] + 'pipeline.log', type='w', name='Pipeline')

    yaml.logging('Starting script with:')
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
    yamlfilename = params['scriptsfolder'] + 'pipeline.parameters.yml'
    copy(yamlfile, yamlfilename)

    # Parameter files
    paramfile_rso = params['scriptsfolder'] + 'remove_small_objects.parameters.yml'
    paramfile_mao = params['scriptsfolder'] + 'merge_adjacent_objects.parameters.yml'
    paramfile_cfi = params['scriptsfolder'] + 'compute_feature_images.parameters.yml'
    paramfile_fbc = params['scriptsfolder'] + 'find_border_contacts.parameters.yml'
    paramfile_pol = params['scriptsfolder'] + 'paths_of_labels.parameters.yml'
    paramfile_pom = params['scriptsfolder'] + 'paths_of_merges.parameters.yml'
    paramfile_fop = params['scriptsfolder'] + 'features_of_paths.parameters.yml'
    paramfile_rdf = params['scriptsfolder'] + 'random_forest.parameters.yml'

    # Copy the script files and parameter files of the pipeline components
    # Copying the parameter file for each pipeline component may seem redundant (it is the same parameter file every
    #   time) but it makes sense if one re-computes part of the pipeline with different parameters. Also it doesn't take
    #   much disk space
    if params['run_remove_small_objects']:
        copy('p161111_00_remove_small_objects.py', params['scriptsfolder'])
        copy(yamlfile, paramfile_rso)
    if params['run_merge_adjacent_objects']:
        copy('p161111_01_merge_adjacent_objects.py', params['scriptsfolder'])
        copy(yamlfile, paramfile_mao)
    if params['run_compute_feature_images']:
        copy('p161111_02_compute_feature_images.py', params['scriptsfolder'])
        copy(yamlfile, paramfile_cfi)
    if params['run_find_border_contacts']:
        copy('p161111_03_find_border_contacts.py', params['scriptsfolder'])
        copy(yamlfile, paramfile_fbc)
    if params['run_paths_of_labels']:
        copy('p161111_04a_paths_of_labels.py', params['scriptsfolder'])
        copy(yamlfile, paramfile_pol)
    if params['run_paths_of_merges']:
        copy('p161111_04b_paths_of_merges.py', params['scriptsfolder'])
        copy(yamlfile, paramfile_pom)
    if params['run_features_of_paths']:
        copy('p161111_05_features_of_paths.py', params['scriptsfolder'])
        copy(yamlfile, paramfile_fop)
    if params['run_random_forest']:
        copy('p161111_06a_random_forest.py', params['scriptsfolder'])
        copy(yamlfile, paramfile_rdf)

    # The pipeline
    # _________________________________________________________________________________________
    if params['run_remove_small_objects']:
        yaml.logging('Removing small objects ...')
        run_remove_small_objects(paramfile_rso)

    if params['run_merge_adjacent_objects']:
        yaml.logging('Merging adjacent objects ...')
        run_merge_adjacent_objects(paramfile_mao)

    if params['run_compute_feature_images']:
        yaml.logging('Computing feature images ...')
        run_compute_feature_images(paramfile_cfi)

    if params['run_find_border_contacts']:
        yaml.logging('Finding border contacts ...')
        run_find_border_contacts(paramfile_fbc)

    if params['run_paths_of_labels']:
        yaml.logging('Calculating paths of labels ...')
        run_paths_of_labels(paramfile_pol)

    if params['run_paths_of_merges']:
        yaml.logging('Calculating paths of merges ...')
        run_paths_of_merges(paramfile_pom)

    if params['run_features_of_paths']:
        yaml.logging('Extracting features along paths ...')
        run_features_of_paths(paramfile_fop)

    if params['run_random_forest']:
        yaml.logging('Running random forst on paths ...')
        run_random_forest(paramfile_rdf)
    # _________________________________________________________________________________________

    yaml.stoplogger()

