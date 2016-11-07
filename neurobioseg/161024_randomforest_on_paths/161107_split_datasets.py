
from hdf5_image_processing import Hdf5ImageProcessingLib as IPL
import processing_lib as lib
import math

__author__ = 'jhennies'


if __name__ == "__main__":

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

    # Sample B
    sample = IPL(
        filepath='/mnt/localdata02/jhennies/neuraldata/cremi_2016/cremi.splB.raw_neurons.crop.h5'
    )

    sample.logging('Sample B datastructure\n---\n{}', sample.datastructure2string())

    reskeys = ('0', '1')
    split_sample = IPL()
    split_sample['z'] = sample.anytask(lib.split, 2, axis=0, result_keys=reskeys, return_only=True)
    split_sample['y'] = sample.anytask(lib.split, 2, axis=1, result_keys=reskeys, return_only=True)
    split_sample['x'] = sample.anytask(lib.split, 2, axis=2, result_keys=reskeys, return_only=True)

    split_sample = split_sample.switch_levels(1, 2)
    sample.logging('Split sample B datastructure\n---\n{}', split_sample.datastructure2string())

    split_sample.write('/mnt/localdata02/jhennies/neuraldata/cremi_2016/cremi.splB.raw_neurons.crop.split_xyz.h5')

    # Sample C
    sample = IPL(
        filepath='/mnt/localdata02/jhennies/neuraldata/cremi_2016/cremi.splC.raw_neurons.crop.h5'
    )

    sample.logging('Sample C datastructure\n---\n{}', sample.datastructure2string())

    reskeys = ('0', '1')
    split_sample = IPL()
    split_sample['z'] = sample.anytask(lib.split, 2, axis=0, result_keys=reskeys, return_only=True)
    split_sample['y'] = sample.anytask(lib.split, 2, axis=1, result_keys=reskeys, return_only=True)
    split_sample['x'] = sample.anytask(lib.split, 2, axis=2, result_keys=reskeys, return_only=True)

    split_sample = split_sample.switch_levels(1, 2)
    sample.logging('Split sample C datastructure\n---\n{}', split_sample.datastructure2string())

    split_sample.write('/mnt/localdata02/jhennies/neuraldata/cremi_2016/cremi.splC.raw_neurons.crop.split_xyz.h5')
