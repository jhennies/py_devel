
from hdf5_processing import Hdf5Processing as h5P
import numpy as np
import pickle
from matplotlib import pyplot as plt

# Locations
path = '/mnt/localdata01/jhennies/neuraldata/results/cremi_2016/170125_plot_undersegm_paths_against_beta/intermed/'
borderctfile = 'cremi.splA.train.borderct.crop.split_x.h5'
gtfile = 'cremi.splA.train.gtlarge.crop.split_x.h5'
segmfile = 'cremi.splA.train.segmlarge.crop.split_x.h5'

# Settings
print_details = True
pickle_result = True
load_from_pickle = True
pickle_file = path + 'object_investigation.pkl'


def get_statistics():

    # Pre-load data structures
    bordercts = h5P()
    bordercts.data_from_file(path + borderctfile, nodata=True)
    gt = h5P()
    gt.data_from_file(path + gtfile, nodata=True)
    segm = h5P()
    segm.data_from_file(path + segmfile, nodata=True)

    bordercts.dss()
    gt.dss()
    segm.dss()

    truepathcount = h5P()
    falsepathcount = h5P()
    mergedobjectslist = h5P()
    nonmergedobjectslist = h5P()

    # Iterate over all segmentation, i.e. betas
    for d, k, cur_segm, kl in segm.data_iterator(leaves_only=True, yield_short_kl=True):

        # print 'Working on kl = {}'.format(kl + [k])

        # Load data
        cur_segm = np.array(cur_segm)
        cur_gt = np.array(gt[kl]['neuron_ids'])

        # Initialize
        truepathcount[kl + [k]] = 0
        falsepathcount[kl + [k]] = 0
        mergedobjectslist[kl + [k]] = []
        nonmergedobjectslist[kl + [k]] = []

        # Iterate over each label in the respective segmentation
        for lbl in np.unique(cur_segm):

            # print '    Checking label {}'.format(lbl)

            # Mask the border contacts to get the pathends of the current object
            cur_bordercts = np.array(bordercts[kl][k]['contacts'])
            cur_bordercts[cur_segm != lbl] = 0

            # Determine the pathends within this label
            indices = np.where(cur_bordercts)
            coords = zip(indices[0], indices[1], indices[2])

            # Make pairwise list of coordinates serving as source and target
            # First determine all pairings
            all_pairs = []
            for i in xrange(0, len(coords) - 1):
                for j in xrange(i + 1, len(coords)):
                    all_pairs.append((coords[i], coords[j]))

            for pair in all_pairs:
                # Determine whether the endpoints are in different gt objects
                if cur_gt[pair[0]] == cur_gt[pair[1]]:
                    # No merge
                    truepathcount[kl][k] += 1
                else:
                    # Merge detected
                    falsepathcount[kl][k] += 1
                    sorted_pair = sorted([cur_gt[pair[0]], cur_gt[pair[1]]])
                    if sorted_pair not in mergedobjectslist[kl][k]:
                        mergedobjectslist[kl][k].append(sorted_pair)
            #             print '        New pair found: {}'.format(sorted_pair)
            # print '        Paths within one object: {}'.format(truepathcount[kl][k])
            # print '        Paths with merge: {}'.format(falsepathcount[kl][k])

        # All labels
        all_labels = np.unique(cur_segm)
        # Those that are involved in a merge
        merged_labels = np.unique(mergedobjectslist[kl][k])
        # Those that are not involved in a merge
        non_merged_labels = list(set(all_labels) - set(merged_labels))

        nonmergedobjectslist[kl][k] = non_merged_labels

        print '++++++++++++++++++++++++++++++++++++++++++++++++++'
        print 'Details for {}'.format(kl + [k])
        print '    Labels not involved in merge:'
        print non_merged_labels
        print '    Labels involved in merge:'
        print merged_labels
        print '++++++++++++++++++++++++++++++++++++++++++++++++++'
        print 'Summary for {}'.format(kl + [k])
        print '    Paths within one object: {}'.format(truepathcount[kl][k])
        print '    {} unmerged objects detected'.format(len(non_merged_labels))
        print '    Paths with merge: {}'.format(falsepathcount[kl][k])
        print '    {} merged objects detected'.format(len(merged_labels))
        print '++++++++++++++++++++++++++++++++++++++++++++++++++'

    result = h5P()
    result['truepathcount'] = truepathcount
    result['falsepathcount'] = falsepathcount
    result['mergedobjectlist'] = mergedobjectslist
    result['nonmergedobjectlist'] = nonmergedobjectslist

    if pickle_result:
        with open(pickle_file, 'wb') as f:
            pickle.dump(result, f)

    return result


def plot_results(result):

    mergedobjectlist = result['mergedobjectlist']
    nonmergedobjectlist = result['nonmergedobjectlist']
    truepathcount = result['truepathcount']
    falsepathcount = result['falsepathcount']

    betas = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

    # plt_handles = range(4)
    # plt_handles2 = dict()

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # for k, v in falsepathcount['x'].iteritems():
    #
    #     false_paths = [v['beta_{}'.format(x)] for x in betas]
    #     true_paths = [truepathcount['x'][k]['beta_{}'.format(x)] for x in betas]
    #
    #     plt_handles[0 + int(k)*2], = ax1.plot(betas, false_paths, label='False paths in {}'.format(k))
    #     plt_handles[1 + int(k)*2], = ax1.plot(betas, true_paths, label='True paths in {}'.format(k))
    #
    #     plt_handles2[k], = ax2.plot(betas, np.array(false_paths).astype(np.float) / np.array(true_paths), label='False/True in {}'.format(k))

    false_paths_0 = [falsepathcount['x', '0']['beta_{}'.format(x)] for x in betas]
    true_paths_0 = [truepathcount['x', '0']['beta_{}'.format(x)] for x in betas]
    false_paths_1 = [falsepathcount['x', '1']['beta_{}'.format(x)] for x in betas]
    true_paths_1 = [truepathcount['x', '1']['beta_{}'.format(x)] for x in betas]

    plot_ax1 = []
    plot_ax1 += ax1.plot(betas, false_paths_0, label='False paths in 0')
    plot_ax1 += ax1.plot(betas, true_paths_0, label='True paths in 0')
    plot_ax1 += ax1.plot(betas, false_paths_1, label='False paths in 1')
    plot_ax1 += ax1.plot(betas, true_paths_1, label='True paths in 1')

    plot_ax2 = []
    plot_ax2 += ax2.plot(betas, np.array(false_paths_0).astype(np.float) / np.array(true_paths_0), label='False/True in 0')
    plot_ax2 += ax2.plot(betas, np.array(false_paths_1).astype(np.float) / np.array(true_paths_1), label='False/True in 1')

    ax1.legend(handles=plot_ax1)
    ax2.legend(handles=plot_ax2)

    plt.show()


if __name__ == '__main__':

    if not load_from_pickle:
        result = get_statistics()
    else:
        with open(pickle_file, 'r') as f:
            result = pickle.load(f)

    plot_results(result)