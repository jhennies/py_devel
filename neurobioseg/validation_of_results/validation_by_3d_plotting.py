
from neuro_seg_plot import NeuroSegPlot as nsp
from hdf5_slim_processing import Hdf5Processing as hp
import numpy as np
import sys

# Parameters:
anisotropy = [10, 1, 1]
interpolation_mode = 'nearest'
transparent = True
opacity = 0.25
# label = '118'
label = 'random'
pathid = '1'
surface_source = 'seg'

# Specify the files
raw_path = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/'
raw_file = 'cremi.splA.train.raw_neurons.crop.split_xyz.h5'
raw_skey = ['z', '0', 'raw']

seg_path = '/mnt/localdata01/jhennies/neuraldata/results/cremi_2016/170208_avoid_redundant_path_calculation_sample_a_slice_z_train01_predict10_full/intermed/'
seg_file = 'cremi.splA.train.segmlarge.crop.split_z.h5'
seg_skey = ['z', '0', 'beta_0.5']

gt_path = seg_path
gt_file = 'cremi.splA.train.gtlarge.crop.split_z.h5'
gt_skey = ['z', '0', 'neuron_ids']

paths_path = seg_path
paths_file = 'cremi.splA.train.paths.crop.split_z.h5'
pathlist_file = 'cremi.splA.train.pathlist.crop.split_z.pkl'
paths_skey = ['z_train0_predict1', 'truepaths', 'z', '0', 'beta_0.5']

# # DEVELOP ----
# # Specify the files
# raw_path = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/'
# raw_file = 'cremi.splA.train.raw_neurons.crop.crop_x10_110_y200_712_z200_712.split_xyz.h5'
# raw_skey = ['x', '1', 'raw']
#
# seg_path = '/mnt/localdata02/jhennies/neuraldata/results/cremi_2016/170124_neurobioseg_x_cropped_avoid_duplicates_develop/intermed/'
# seg_file = 'cremi.splA.train.segmlarge.crop.crop_x10_110_y200_712_z200_712.split_x.h5'
# seg_skey = ['x', '1', 'beta_0.5']
#
# gt_path = seg_path
# gt_file = 'cremi.splA.train.gtlarge.crop.crop_x10_110_y200_712_z200_712.split_x.h5'
# gt_skey = ['x', '1', 'neuron_ids']
#
# paths_path = '/mnt/localdata02/jhennies/neuraldata/results/cremi_2016/170124_neurobioseg_x_cropped_avoid_duplicates_develop/intermed/'
# paths_file = 'cremi.splA.train.paths.crop.crop_x10_110_y200_712_z200_712.split_x.h5'
# pathlist_file = 'cremi.splA.train.pathlist.crop.crop_x10_110_y200_712_z200_712.split_x.pkl'
# paths_skey = ['predict', 'falsepaths', 'x', '1', 'beta_0.5']

# crop = np.s_[0:10, 0:100, 0:100]

# Load path
paths = hp(filepath=paths_path + paths_file, nodata=True, skeys=[paths_skey])[paths_skey]
print paths.keys()
if label == 'random':
    paths_list = []
    for d, k, v, kl in paths.data_iterator(leaves_only=True):
        paths_list.append(kl)
    import random
    chosen_path = random.choice(paths_list)
    label = chosen_path[0]
    pathid = chosen_path[1]
# label = paths.keys()[1]
print 'Selected label = {}'.format(label)
print 'Selected pathid = {}'.format(pathid)
path = np.array(paths[label, pathid])

sys.exit()

import processing_lib as lib

if surface_source == 'seg':
    # Load segmentation
    seg_image = np.array(hp(filepath=seg_path + seg_file, nodata=True, skeys=[seg_skey])[seg_skey])
    seg_image[seg_image != int(label)] = 0

elif surface_source == 'gt':
    # Load ground truth
    seg_image = np.array(hp(filepath=gt_path + gt_file, nodata=True, skeys=[gt_skey])[gt_skey])
    gt_labels = np.unique(lib.getvaluesfromcoords(seg_image, path))
    t_seg_image = np.array(seg_image)
    for l in gt_labels:
        t_seg_image[t_seg_image == l] = 0
    seg_image[t_seg_image > 0] = 0
    t_seg_image = None

else:
    sys.exit()

crop = lib.find_bounding_rect(seg_image, s_=True)
print 'crop = {}'.format(crop)
seg_image = lib.crop_bounding_rect(seg_image, crop)

path = np.swapaxes(path, 0, 1)
path[0] = path[0] - crop[0].start
path[1] = path[1] - crop[1].start
path[2] = path[2] - crop[2].start

# Load raw image
raw_image = np.array(hp(filepath=raw_path + raw_file, nodata=True, skeys=[raw_skey])[raw_skey])

raw_image = lib.crop_bounding_rect(raw_image, crop)

nsp.start_figure()
nsp.add_path(path, anisotropy=anisotropy)
nsp.add_iso_surfaces(seg_image, anisotropy=anisotropy, colormap='Spectral')
nsp.add_xyz_planes(raw_image, anisotropy=anisotropy)
nsp.show()

# plot.path_in_segmentation_data_bg(raw_image, seg_image, path)

# plot.show()
