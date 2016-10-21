
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


def merge_adjacent_objects(hfp):

    params = hfp.get_params()
    thisparams = params['merge_adjacent_objects']

    numberbysize = thisparams['numberbysize']
    numberbyrandom = thisparams['numberbyrandom']
    targetnames = params['largeobjmnames']

    # Get only the relevant labels
    labels = lib.unique(hfp['largeobj'])
    hfp.logging('labels = {}', labels)

    # Seed the randomize function
    random.seed(thisparams['seed'])

    hfp.astype(np.uint32, keys='largeobj')
    (grag, rag) = graphs.gridRegionAdjacencyGraph(hfp['largeobj'], ignoreLabel=0)
    edge_ids = rag.edgeIds()
    # hfp.logging('Edge ids: {}', edge_ids)

    # Type 1:
    # Select edges by size (smallest edges)
    hfp.logging('Number of edgeLengths = {}', len(rag.edgeLengths()))
    edgelen_ids = dict(zip(edge_ids, rag.edgeLengths()))
    # ifp.logging('edgelen_ids = {}', edgelen_ids)
    sorted_edgelens = np.sort(rag.edgeLengths())
    #
    smallest_merge_lens = sorted_edgelens[0:numberbysize]
    hfp.logging('Lengths selected for merging: {}', smallest_merge_lens)
    #
    smallest_merge_ids = []
    for x in smallest_merge_lens:
        edge_id = edgelen_ids.keys()[edgelen_ids.values().index(x)]
        smallest_merge_ids.append(edge_id)
        edgelen_ids.pop(edge_id)
    #
    edge_ids = edgelen_ids.keys()
    hfp.logging('Edge IDs selected for merging due to size: {}', smallest_merge_ids)

    # Type 2:
    # Randomly choose edges
    random_merge_ids = random.sample(edge_ids, numberbyrandom)
    hfp.logging('Edge IDs randomly selected for merging: {}', random_merge_ids)

    # Now get the label ids
    smallest_merge_labelids_u = [rag.uId(rag.edgeFromId(x)) for x in smallest_merge_ids]
    smallest_merge_labelids_v = [rag.vId(rag.edgeFromId(x)) for x in smallest_merge_ids]
    smallest_merge_labelids = list(zip(smallest_merge_labelids_u, smallest_merge_labelids_v))
    random_merge_labelids_u = [rag.uId(rag.edgeFromId(x)) for x in random_merge_ids]
    random_merge_labelids_v = [rag.vId(rag.edgeFromId(x)) for x in random_merge_ids]
    random_merge_labelids = list(zip(random_merge_labelids_u, random_merge_labelids_v))
    hfp.logging('Label IDs selected for merging by size: {}', smallest_merge_labelids)
    hfp.logging('Label IDs randomly selected for merging: {}', random_merge_labelids)

    # Concatenate
    all_merge_labelids = smallest_merge_labelids + random_merge_labelids
    # Sort
    hfp.logging('all_merge_labelids = {}', all_merge_labelids)
    all_merge_labelids = [sorted(x) for x in all_merge_labelids]
    all_merge_labelids = sorted(all_merge_labelids)
    hfp.logging('all_merge_labelids = {}', all_merge_labelids)

    # Store this for later use
    hfp[targetnames[1]] = smallest_merge_labelids
    hfp[targetnames[2]] = random_merge_labelids
    hfp[targetnames[3]] = all_merge_labelids

    # Create change hash list
    change_hash = IPL(data=dict(zip(np.unique(all_merge_labelids), [[x,] for x in np.unique(all_merge_labelids)])))
    for i in xrange(0, 3):
        prev_change_hash = IPL(data=change_hash)
        for x in all_merge_labelids:
            hfp.logging('Adding {} and {}', *x)
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
    # And now we have a perfect change list which we just need to iterate over and change the labels in the image
    hfp.logging('change_hash after change:')
    hfp.logging(change_hash)
    hfp[targetnames[4]] = change_hash

    # Create the merged image
    # hfp.deepcopy_entry('largeobj', targetnames[0])
    hfp.rename_entry('largeobj', targetnames[0])
    for k, v in change_hash.iteritems():
        for x in v:
            if x != k:
                hfp.logging('Setting {} to {}!', x, k)
                hfp.filter_values(x, type='eq', setto=k, keys=targetnames[0])


if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'

    hfp = IPL(
        yaml=yamlfile,
        yamlspec={'path': 'datafolder', 'filename': 'largeobjfile', 'skeys': 'largeobjname'},
        tkeys='largeobj',
        castkey=None
    )
    params = hfp.get_params()
    hfp.startlogger(filename=params['resultfolder'] + 'merge_adjacent_objects.log', type='w')

    try:

        # Copy the script file and the parameters to the scriptsfolder
        copy(inspect.stack()[0][1], params['scriptsfolder'])
        copy(yamlfile, params['scriptsfolder'] + 'merge_adjacent_objects.parameters.yml')
        # Write script and parameters to the logfile
        hfp.code2log(inspect.stack()[0][1])
        hfp.logging('')
        hfp.yaml2log()
        hfp.logging('')

        hfp.logging('\nhfp datastructure: \n\n{}', hfp.datastructure2string(maxdepth=1))

        merge_adjacent_objects(hfp)

        hfp.write(filepath=params['intermedfolder'] + params['largeobjmfile'])

        hfp.logging('\nFinal dictionary structure:\n---\n{}', hfp.datastructure2string())
        hfp.logging('')
        hfp.stoplogger()

    except:

        hfp.errout('Unexpected error')
