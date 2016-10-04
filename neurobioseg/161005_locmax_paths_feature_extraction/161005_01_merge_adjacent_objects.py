
from image_processing import ImageFileProcessing
import random
import vigra.graphs as graphs
import numpy as np
import os
import inspect

__author__ = 'jhennies'

if __name__ == '__main__':

    # numberbysize = 10
    # numberbyrandom = 10
    #
    # datafolder = '/media/julian/Daten/neuraldata/cremi_2016/'
    # file = 'cremi.splA.raw_neurons.crop.crop_10-200-200_110-712-712.largeobjects.h5'
    # names = ('largeobjects',)
    # keys = ('labels',)
    # targetfolder = 'cremi.splA.raw_neurons.crop.crop_10-200-200_110-712-712_merges/'

    # ifp = ImageFileProcessing(
    #     datafolder,
    #     file, asdict=True,
    #     image_names=names,
    #     keys=keys)

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'

    ifp = ImageFileProcessing(
        yaml=yamlfile,
        yamlspec={'image_path': 'intermedfolder', 'image_file': 'labelfile', 'image_names': 'labelname'},
        asdict=True,
        keys=('labels',)
    )

    ifp.startlogger(filename=ifp.get_params()['intermedfolder'] + 'merge_adjacent_objects.log', type='a')

    ifp.code2log(__file__)
    ifp.logging('')

    ifp.logging('yamlfile = {}', yamlfile)
    ifp.logging('ifp.get_data().keys() = {}', ifp.get_data().keys())

    numberbysize = ifp.get_params()['merge_adjacent_objects']['numberbysize']
    numberbyrandom = ifp.get_params()['merge_adjacent_objects']['numberbyrandom']
    targetfolder = ifp.get_params()['intermedfolder']

    # Done: Randomly select an edge
    # Done: Find adjacent objects
    # Done: Extract both adjacent objects as an image (needed for subsequent learning)
    # TODO: Merge the objects
    # Done: Select edges by size? Small edges are more relevant for our algorithm

    # # Find all relevant labels
    # ifp.addfromfile('{}largeobjects.h5'.format(folder), image_ids=(0,))

    ifp.logging('keys = {}', ifp.get_data().keys())

    # # This get all the labels
    # labels = ifp.unique(ids='labels')

    # Get only the relevant labels
    labels = ifp.unique(ids='labels')
    ifp.logging('labels = {}', labels)

    # Seed the randomize function
    random.seed(1)

    ifp.astype(np.uint32, ids='labels')
    (grag, rag) = ifp.anytask_rtrn(graphs.gridRegionAdjacencyGraph, ignoreLabel=0, ids='labels')
    edge_ids = rag.edgeIds()
    ifp.logging('Edge ids: {}', edge_ids)

    # Type 1:
    # Select edges by size (smallest edges)
    ifp.logging('Number of edgeLengths = {}', len(rag.edgeLengths()))
    edgelen_ids = dict(zip(edge_ids, rag.edgeLengths()))
    ifp.logging('edgelen_ids = {}', edgelen_ids)
    sorted_edgelens = np.sort(rag.edgeLengths())
    #
    smallest_merge_lens = sorted_edgelens[0:numberbysize]
    ifp.logging('Lengths selected for merging: {}', smallest_merge_lens)
    #
    smallest_merge_ids = []
    for x in smallest_merge_lens:
        edge_id = edgelen_ids.keys()[edgelen_ids.values().index(x)]
        smallest_merge_ids.append(edge_id)
        edgelen_ids.pop(edge_id)
    #
    edge_ids = edgelen_ids.keys()
    ifp.logging('Edge IDs selected for merging due to size: {}', smallest_merge_ids)

    # Type 2:
    # Randomly choose edges
    random_merge_ids = random.sample(edge_ids, numberbyrandom)
    ifp.logging('Edge IDs randomly selected for merging: {}', random_merge_ids)

    # Now get the label ids
    smallest_merge_labelids_u = [rag.uId(rag.edgeFromId(x)) for x in smallest_merge_ids]
    smallest_merge_labelids_v = [rag.vId(rag.edgeFromId(x)) for x in smallest_merge_ids]
    smallest_merge_labelids = list(zip(smallest_merge_labelids_u, smallest_merge_labelids_v))
    random_merge_labelids_u = [rag.uId(rag.edgeFromId(x)) for x in random_merge_ids]
    random_merge_labelids_v = [rag.vId(rag.edgeFromId(x)) for x in random_merge_ids]
    random_merge_labelids = list(zip(random_merge_labelids_u, random_merge_labelids_v))
    ifp.logging('Label IDs selected for merging by size: {}', smallest_merge_labelids)
    ifp.logging('Label IDs randomly selected for merging: {}', random_merge_labelids)

    # Concatenate
    all_merge_labelids = smallest_merge_labelids + random_merge_labelids

    # Store this for later use
    ifp.set_data_dict({'mergeids_small': smallest_merge_labelids, 'mergeids_random': random_merge_labelids, 'mergeids_all': all_merge_labelids}, append=True)
    # ifp.write(filepath=targetfolder + 'mergeids.h5', ids=('mergesmall', 'mergerandom', 'mergeall'))

    # Extract merged image
    ifp.deepcopy_entry('labels', 'labels_merged')
    for x in all_merge_labelids:
        ifp.filter_values(x[1], type='eq', setto=x[0], ids='labels_merged')
    #     ifp.getlabel(x, ids='labels', targetids='labels_merged_{}_{}'.format(x[0], x[1]))
    # ifp.getlabel(tuple(np.unique(all_merge_labelids)), ids='labels', targetids='get_labels_merged')
    ifp.write(filepath=targetfolder + 'labels_merged.h5', ids=('labels_merged', 'mergeids_small', 'mergeids_random', 'mergeids_all'))
    # ifp.write(filepath=targetfolder + 'all_images.h5')

    ifp.logging('')
    ifp.stoplogger()