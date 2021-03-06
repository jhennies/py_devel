
from hdf5_image_processing import Hdf5ImageProcessingLib as IPL
import processing_lib as lib

# Sample A probs
probs_a = IPL(
    filepath='/mnt/localdata01/jhennies/neuraldata/cremi_2016/sample_A_train_betas/sample_A_train_mcseg_beta_0.5.h5'
)

probs_a.logging('Probs A datastructure\n---\n{}', probs_a.datastructure2string())

probs_a.anytask(lib.swapaxes, 0, 2)

probs_a.write('/mnt/localdata01/jhennies/neuraldata/cremi_2016/sample_A_train_betas/cremi.splA.train.seg_beta_0.5.crop.h5')

reskeys = ('0', '1')
split_probs_a = IPL()
split_probs_a['z'] = probs_a.anytask(lib.split, 2, axis=0, result_keys=reskeys, return_only=True, rtrntype=IPL)
split_probs_a['y'] = probs_a.anytask(lib.split, 2, axis=1, result_keys=reskeys, return_only=True, rtrntype=IPL)
split_probs_a['x'] = probs_a.anytask(lib.split, 2, axis=2, result_keys=reskeys, return_only=True, rtrntype=IPL)

split_probs_a = split_probs_a.switch_levels(1, 2)
probs_a.logging('Split sample A datastructure\n---\n{}', split_probs_a.datastructure2string())

split_probs_a.write('/mnt/localdata01/jhennies/neuraldata/cremi_2016/sample_A_train_betas/cremi.splA.train.seg_beta_0.5.crop.split_xyz.h5')
