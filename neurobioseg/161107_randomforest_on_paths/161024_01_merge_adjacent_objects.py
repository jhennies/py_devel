
# from image_processing import ImageFileProcessing
from hdf5_image_processing import Hdf5ImageProcessingLib as IPL
import random
import vigra.graphs as graphs
import numpy as np
import os
import inspect
from shutil import copy
import processing_lib as lib
from copy import deepcopy

__author__ = 'jhennies'


def merge_adjacent_objects(ipl, key, thisparams):
    """
    :param ipl:

    ipl.get_params():

        merge_adjacent_objects
            seed
            numberbysize
            numberbyrandom

        largeobjmnames
            - 'largeobj_merged'
            - 'mergeids_small'
            - 'mergeids_random'
            - 'mergeids_all'
            - 'change_hash'

    :param key: the source key for calculation
    """

    numberbysize = thisparams['numberbysize']
    numberbyrandom = thisparams['numberbyrandom']
    targetnames = params['largeobjmnames']

    # Get only the relevant labels
    labels = lib.unique(ipl[key])
    ipl.logging('labels = {}', labels)

    # Seed the randomize function
    random.seed(thisparams['seed'])

    ipl.astype(np.uint32, keys=key)
    (grag, rag) = graphs.gridRegionAdjacencyGraph(ipl[key], ignoreLabel=0)
    edge_ids = rag.edgeIds()
    # ipl.logging('Edge ids: {}', edge_ids)

    # Type 1:
    # Select edges by size (smallest edges)
    ipl.logging('Number of edgeLengths = {}', len(rag.edgeLengths()))
    edgelen_ids = dict(zip(edge_ids, rag.edgeLengths()))
    # ifp.logging('edgelen_ids = {}', edgelen_ids)
    sorted_edgelens = np.sort(rag.edgeLengths())
    #
    smallest_merge_lens = sorted_edgelens[0:numberbysize]
    ipl.logging('Lengths selected for merging: {}', smallest_merge_lens)
    #
    smallest_merge_ids = []
    for x in smallest_merge_lens:
        edge_id = edgelen_ids.keys()[edgelen_ids.values().index(x)]
        smallest_merge_ids.append(edge_id)
        edgelen_ids.pop(edge_id)
    #
    edge_ids = edgelen_ids.keys()
    ipl.logging('Edge IDs selected for merging due to size: {}', smallest_merge_ids)

    # Type 2:
    # Randomly choose edges
    random_merge_ids = random.sample(edge_ids, numberbyrandom)
    ipl.logging('Edge IDs randomly selected for merging: {}', random_merge_ids)

    # Now get the label ids
    smallest_merge_labelids_u = [rag.uId(rag.edgeFromId(x)) for x in smallest_merge_ids]
    smallest_merge_labelids_v = [rag.vId(rag.edgeFromId(x)) for x in smallest_merge_ids]
    smallest_merge_labelids = list(zip(smallest_merge_labelids_u, smallest_merge_labelids_v))
    random_merge_labelids_u = [rag.uId(rag.edgeFromId(x)) for x in random_merge_ids]
    random_merge_labelids_v = [rag.vId(rag.edgeFromId(x)) for x in random_merge_ids]
    random_merge_labelids = list(zip(random_merge_labelids_u, random_merge_labelids_v))
    ipl.logging('Label IDs selected for merging by size: {}', smallest_merge_labelids)
    ipl.logging('Label IDs randomly selected for merging: {}', random_merge_labelids)

    # Concatenate
    all_merge_labelids = smallest_merge_labelids + random_merge_labelids
    # Sort
    ipl.logging('all_merge_labelids = {}', all_merge_labelids)
    all_merge_labelids = [sorted(x) for x in all_merge_labelids]
    all_merge_labelids = sorted(all_merge_labelids)
    ipl.logging('all_merge_labelids = {}', all_merge_labelids)

    # Store this for later use
    ipl[targetnames[1]] = smallest_merge_labelids
    ipl[targetnames[2]] = random_merge_labelids
    ipl[targetnames[3]] = all_merge_labelids

    # Create change hash list
    change_hash = IPL(data=dict(zip(np.unique(all_merge_labelids), [[x,] for x in np.unique(all_merge_labelids)])))
    for i in xrange(0, 3):
        prev_change_hash = IPL(data=change_hash)
        for x in all_merge_labelids:
            ipl.logging('Adding {} and {}', *x)
            change_hash[x[0]] += change_hash[x[1]]
            change_hash[x[0]] = list(np.unique(change_hash[x[0]]))
            change_hash[x[1]] += change_hash[x[0]]
            change_hash[x[1]] = list(np.unique(change_hash[x[1]]))
    # This removes the redundancy from the hash
    def reduce(hash):
        br = False
        for k, v in hash.iteritems():
            for x in v:
                if x != k:
                    if x in hash.keys():
                        del hash[x]
                        reduce(hash)
                        br = True
                        break
                    else:
                        br = False
            if br:
                break
    reduce(change_hash)
    # Change the list in the hash to np-arrays for better storage in h5 files
    for k, v in change_hash.iteritems():
        change_hash[k] = np.array(v)
    # And now we have a perfect change list which we just need to iterate over and change the labels in the image
    ipl.logging('change_hash after change:')
    ipl.logging(change_hash)
    ipl[targetnames[4]] = change_hash

    # Create the merged image
    # ipl.deepcopy_entry('largeobj', targetnames[0])
    ipl.rename_entry(key, targetnames[0])
    for k, v in change_hash.iteritems():
        for x in v:
            if x != k:
                ipl.logging('Setting {} to {}!', x, k)
                ipl.filter_values(x, type='eq', setto=k, keys=targetnames[0])

    return ipl


def merge_adjacent_objects_image_iteration(ipl):

    params = ipl.get_params()
    thisparams = params['merge_adjacent_objects']

    merged = IPL()

    for d, k, v, kl in ipl.data_iterator(yield_short_kl=True):

        ipl.logging('Working on image: {}', kl + [k])

        if k == params['largeobjname']:

            data = IPL(data={k: v})
            data.setlogger(ipl.getlogger())
            merged[kl] = merge_adjacent_objects(data, k, thisparams)

    return merged

if __name__ == '__main__':

    resultsfolder = '/mnt/localdata02/jhennies/neuraldata/results/cremi_2016/161107_random_forest_of_paths/'

    yamlfile = resultsfolder + '/parameters.yml'

    ipl = IPL(
        yaml=yamlfile,
        yamlspec={'path': 'intermedfolder', 'filename': 'largeobjfile'}
    )
    params = ipl.get_params()
    ipl.startlogger(filename=params['resultfolder'] + 'merge_adjacent_objects.log', type='w')

    try:

        # Copy the script file and the parameters to the scriptsfolder
        copy(inspect.stack()[0][1], params['scriptsfolder'])
        copy(yamlfile, params['scriptsfolder'] + 'merge_adjacent_objects.parameters.yml')
        # Write script and parameters to the logfile
        ipl.code2log(inspect.stack()[0][1])
        ipl.logging('')
        ipl.yaml2log()
        ipl.logging('')

        ipl.logging('\nipl datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

        merged = merge_adjacent_objects_image_iteration(ipl)

        merged.write(filepath=params['intermedfolder'] + params['largeobjmfile'])

        ipl.logging('\nFinal dictionary structure:\n---\n{}', merged.datastructure2string())
        ipl.logging('')
        ipl.stoplogger()

    except:

        ipl.errout('Unexpected error')
