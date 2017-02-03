
from hdf5_image_processing import Hdf5ImageProcessingLib as ipl
import processing_lib as lib
import numpy as np


def split_in_xyz(data, dims):

    reskeys = ('0', '1')
    split_data = ipl()

    if 'z' in dims:
        split_data['z'] = data.anytask(lib.split, 2, axis=0, result_keys=reskeys, return_only=True, rtrntype=ipl)
    if 'y' in dims:
        split_data['y'] = data.anytask(lib.split, 2, axis=1, result_keys=reskeys, return_only=True, rtrntype=ipl)
    if 'x' in dims:
        split_data['x'] = data.anytask(lib.split, 2, axis=2, result_keys=reskeys, return_only=True, rtrntype=ipl)

    split_data = split_data.switch_levels(1, 2)

    return split_data


debug = False

path = '/mnt/localdata01/jhennies/neuraldata/cremi_2016/sample_BC_train_betas/'

# Set sourcenames
sourcename = 'cremi.splC.train.mcseg_beta_{}.crop.h5'
ids = ['0.3', '0.35', '0.4', '0.45', '0.5', '0.55', '0.6', '0.65', '0.7']

# Set targetnames
targetname = 'cremi.splC.train.mcseg_beta_{}.crop{}.h5'
append2name = '.split_xyz'

# Generate file name lists
sourcefiles = [sourcename.format(x) for x in ids]
print 'files = {}'.format(sourcefiles)
targetfiles = [targetname.format(x, append2name) for x in ids]
print 'targetfiles = {}'.format(targetfiles)

# Parameters
dims = ['x', 'y', 'z']

files = dict(zip(sourcefiles, targetfiles))

for source, target in files.iteritems():

    sourcefilepath = path + source
    print 'sourcefilepath = {}'.format(sourcefilepath)

    targetfilepath = path + target
    print 'targetfilepath = {}'.format(targetfilepath)

    if not debug:

        # Load data
        data = ipl(filepath=sourcefilepath, nodata=False)

        # Print shape
        def shp(x):
            return x.shape
        print 'data.dss(function=shp) = {}'.format(data.dss(function=shp))

        # Split
        data = split_in_xyz(data, dims)

        print 'data.dss(function=shp) = {}'.format(data.dss(function=shp))

        # Write to target
        data.write(filepath=targetfilepath)




