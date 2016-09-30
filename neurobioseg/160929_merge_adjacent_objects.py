
from image_processing import ImageFileProcessing
import random
import vigra.graphs as graphs
import numpy as np

__author__ = 'jhennies'

if __name__ == '__main__':

    numberbysize = 10
    numberbyrandom = 5

    folder = '/media/julian/Daten/neuraldata/cremi_2016/'
    file = 'cremi.splA.raw_neurons.crop.crop_10-200-200_110-712-712.largeobjects.h5'
    names = ('largeobjects',)
    keys = ('labels',)
    targetfolder = 'cremi.splA.raw_neurons.crop.crop_10-200-200_110-712-712_merges/'

    ifp = ImageFileProcessing(
        folder,
        file, asdict=True,
        image_names=names,
        keys=keys)

    ifp.startlogger(filename=None, type='a')

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
    # labels = ifp.get_image('largeobjects')
    labels = ifp.unique(ids='labels')
    ifp.logging('labels = {}', labels)

    # Seed the randomize function
    random.seed(1)

    # # Choose one label
    # lbl = random.choice(labels)
    # ifp.logging('Choice: {}', lbl)
    #
    # ifp.astype(np.uint32, ids='labels')
    # (grag, rag) = ifp.anytask_rtrn(graphs.gridRegionAdjacencyGraph, ignoreLabel=0, ids='labels')
    # ifp.logging('RAG: {}', rag)
    #
    # ifp.logging('Node ids: {}', rag.nodeIds())

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

    # Store this for later use
    ifp.set_data_dict({'mergesmall': smallest_merge_labelids, 'mergerandom': random_merge_labelids}, append=True)
    ifp.write(filename=targetfolder + 'mergeids.h5', ids=('mergesmall', 'mergerandom'))

    # # Detect the labels at either side of the merge edge
    # u = rag.u(merge_edge)
    # v = rag.v(merge_edge)
    # uid = rag.id(u)
    # vid = rag.id(v)
    # ifp.logging('Merging u = {} and v = {}', uid, vid)
    #
    # # Extract both candidates as own image (needed as ground truth for random forest training)
    # ifp.getlabel((uid, vid), ids='labels', targetids='merged')
    # # ifp.write(filename='cremi.splA.raw_neurons.crop.crop_10-200-200_110-712-712/merged_{}_{}.h5'.format(uid, vid), ids=('merged',))
    #
    # # Merge
    # ifp.filter_values(vid, type='eq', setto=uid, ids='labels', targetids='labels_merged')
    # # ifp.write(filename='cremi.splA.raw_neurons.crop.crop_10-200-200_110-712-712/labels_merged.h5', ids=('labels', 'labels_merged'))

    ifp.logging('')
    ifp.stoplogger()