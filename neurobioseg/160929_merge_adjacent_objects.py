
from image_processing import ImageFileProcessing
import random
import vigra.graphs as graphs
import numpy as np

__author__ = 'jhennies'

if __name__ == '__main__':

    folder = '/media/julian/Daten/neuraldata/cremi_2016/'
    file = 'cremi.splA.raw_neurons.crop.crop_10-200-200_110-712-712.largeobjects.h5'
    names = ('largeobjects',)
    keys = ('labels',)
    resultfolder = 'develop/160927_determine_shortest_paths/160929_pthsovrdist_pow10/'

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
    # TODO: Select edges by size? Small edges are more relevant for our algorithm

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

    # Randomly choose an edge
    ifp.astype(np.uint32, ids='labels')
    (grag, rag) = ifp.anytask_rtrn(graphs.gridRegionAdjacencyGraph, ignoreLabel=0, ids='labels')
    merge_edge = rag.edgeFromId(random.choice(rag.edgeIds()))
    ifp.logging('Edge ids: {}', rag.edgeIds())
    ifp.logging('Merge edge: {}', rag.id(merge_edge))

    # Detect the labels at either side of the merge edge
    u = rag.u(merge_edge)
    v = rag.v(merge_edge)
    uid = rag.id(u)
    vid = rag.id(v)
    ifp.logging('Merging u = {} and v = {}', uid, vid)

    # Extract both candidates as own image (needed as ground truth for random forest training)
    ifp.getlabel((uid, vid), ids='labels', targetids='merged')
    ifp.write(filename='cremi.splA.raw_neurons.crop.crop_10-200-200_110-712-712/merged_{}_{}.h5'.format(uid, vid), ids=('merged',))

    # Merge
    ifp.filter_values(vid, type='eq', setto=uid, ids='labels', targetids='labels_merged')
    ifp.write(filename='cremi.splA.raw_neurons.crop.crop_10-200-200_110-712-712/labels_merged.h5', ids=('labels', 'labels_merged'))

    ifp.logging('')
    ifp.stoplogger()