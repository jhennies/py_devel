
from hdf5_image_processing import Hdf5ImageProcessingLib as IPL
import processing_lib as lib
import math
import numpy as np

__author__ = 'jhennies'

def split_in_xyz(ipl):

    ipl.logging('Datastructure\n---\n{}', ipl.datastructure2string())

    reskeys = ('0', '1')
    ipl_split = IPL()
    ipl_split['z'] = ipl.anytask(lib.split, 2, axis=0, result_keys=reskeys, return_only=True, rtrntype=IPL)
    ipl_split['y'] = ipl.anytask(lib.split, 2, axis=1, result_keys=reskeys, return_only=True, rtrntype=IPL)
    ipl_split['x'] = ipl.anytask(lib.split, 2, axis=2, result_keys=reskeys, return_only=True, rtrntype=IPL)

    ipl_split = ipl_split.switch_levels(1, 2)
    ipl.logging('Split sample datastructure\n---\n{}', ipl_split.datastructure2string())

    return ipl_split


if __name__ == "__main__":

    infiles = [
        '/mnt/localdata02/jhennies/neuraldata/cremi_2016/cremi.splA.train.probs.crop.h5',
        '/mnt/localdata02/jhennies/neuraldata/cremi_2016/cremi.splA.train.raw_neurons.crop.h5'
    ]
    outfiles = [
        '/mnt/localdata02/jhennies/neuraldata/cremi_2016/cremi.splA.train.probs.crop.crop_x10_110_y200_712_z200_712.split_xyz.h5',
        '/mnt/localdata02/jhennies/neuraldata/cremi_2016/cremi.splA.train.raw_neurons.crop.crop_x10_110_y200_712_z200_712.split_xyz.h5'
    ]

    for i in xrange(0, len(infiles)):

        ipl = IPL(
            filepath=infiles[i]
        )
        ipl.logging('Datastructure\n---\n{}', ipl.datastructure2string())

        ipl.crop_bounding_rect(bounds=np.s_[10:110, 200:712, 200:712])

        def shape(image):
            return image.shape
        print ipl.datastructure2string(function=shape)

        ipl_split = split_in_xyz(ipl)

        ipl_split.write(filepath=outfiles[i])

    # # Sample A
    # sample_a = IPL(
    #     filepath='/mnt/localdata02/jhennies/neuraldata/cremi_2016/cremi.splA.raw_neurons.crop.h5'
    # )
    #
    # sample_a.logging('Sample A datastructure\n---\n{}', sample_a.datastructure2string())
    #
    # reskeys = ('0', '1')
    # split_sample_a = IPL()
    # split_sample_a['z'] = sample_a.anytask(lib.split, 2, axis=0, result_keys=reskeys, return_only=True)
    # split_sample_a['y'] = sample_a.anytask(lib.split, 2, axis=1, result_keys=reskeys, return_only=True)
    # split_sample_a['x'] = sample_a.anytask(lib.split, 2, axis=2, result_keys=reskeys, return_only=True)
    #
    # split_sample_a = split_sample_a.switch_levels(1, 2)
    # sample_a.logging('Split sample A datastructure\n---\n{}', split_sample_a.datastructure2string())
    #
    # split_sample_a.write('/mnt/localdata02/jhennies/neuraldata/cremi_2016/cremi.splA.raw_neurons.crop.split_xyz.h5')

    # # Sample B
    # sample = IPL(
    #     filepath='/mnt/localdata02/jhennies/neuraldata/cremi_2016/cremi.splB.raw_neurons.crop.h5'
    # )
    #
    # sample.logging('Sample B datastructure\n---\n{}', sample.datastructure2string())
    #
    # reskeys = ('0', '1')
    # split_sample = IPL()
    # split_sample['z'] = sample.anytask(lib.split, 2, axis=0, result_keys=reskeys, return_only=True)
    # split_sample['y'] = sample.anytask(lib.split, 2, axis=1, result_keys=reskeys, return_only=True)
    # split_sample['x'] = sample.anytask(lib.split, 2, axis=2, result_keys=reskeys, return_only=True)
    #
    # split_sample = split_sample.switch_levels(1, 2)
    # sample.logging('Split sample B datastructure\n---\n{}', split_sample.datastructure2string())
    #
    # split_sample.write('/mnt/localdata02/jhennies/neuraldata/cremi_2016/cremi.splB.raw_neurons.crop.split_xyz.h5')
    #
    # # Sample C
    # sample = IPL(
    #     filepath='/mnt/localdata02/jhennies/neuraldata/cremi_2016/cremi.splC.raw_neurons.crop.h5'
    # )
    #
    # sample.logging('Sample C datastructure\n---\n{}', sample.datastructure2string())
    #
    # reskeys = ('0', '1')
    # split_sample = IPL()
    # split_sample['z'] = sample.anytask(lib.split, 2, axis=0, result_keys=reskeys, return_only=True)
    # split_sample['y'] = sample.anytask(lib.split, 2, axis=1, result_keys=reskeys, return_only=True)
    # split_sample['x'] = sample.anytask(lib.split, 2, axis=2, result_keys=reskeys, return_only=True)
    #
    # split_sample = split_sample.switch_levels(1, 2)
    # sample.logging('Split sample C datastructure\n---\n{}', split_sample.datastructure2string())
    #
    # split_sample.write('/mnt/localdata02/jhennies/neuraldata/cremi_2016/cremi.splC.raw_neurons.crop.split_xyz.h5')

    # # Sample A probs
    # probs_a = IPL(
    #     filepath='/mnt/localdata02/jhennies/neuraldata/cremi_2016/cremi.splA.probs_cantorV1.h5'
    # )
    #
    # probs_a.logging('Probs A datastructure\n---\n{}', probs_a.datastructure2string())
    #
    # reskeys = ('0', '1')
    # split_probs_a = IPL()
    # split_probs_a['z'] = probs_a.anytask(lib.split, 2, axis=0, result_keys=reskeys, return_only=True, rtrntype=IPL)
    # split_probs_a['y'] = probs_a.anytask(lib.split, 2, axis=1, result_keys=reskeys, return_only=True, rtrntype=IPL)
    # split_probs_a['x'] = probs_a.anytask(lib.split, 2, axis=2, result_keys=reskeys, return_only=True, rtrntype=IPL)
    #
    # split_probs_a = split_probs_a.switch_levels(1, 2)
    # probs_a.logging('Split sample A datastructure\n---\n{}', split_probs_a.datastructure2string())
    #
    # split_probs_a.write('/mnt/localdata02/jhennies/neuraldata/cremi_2016/cremi.splA.probs.crop.split_xyz.h5')
