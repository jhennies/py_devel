
from hdf5_image_processing import Hdf5ImageProcessingLib as ipl
import processing_lib as lib
import numpy as np

debug = False

path = '/mnt/localdata01/jhennies/neuraldata/cremi_2016/sample_A_train_betas/'

# Set sourcenames
sourcename = 'cremi.splA.train.mcseg_beta_{}.crop.h5'
ids = ['0.3', '0.35', '0.4', '0.45', '0.5', '0.55', '0.6', '0.65', '0.7']

# Set targetnames
targetname = 'cremi.splA.train.mcseg_beta_{}.crop{}.h5'
append2name = '.crop_x10_110_y200_712_z200_712'

# Generate file name lists
sourcefiles = [sourcename.format(x) for x in ids]
print 'files = {}'.format(sourcefiles)
targetfiles = [targetname.format(x, append2name) for x in ids]
print 'targetfiles = {}'.format(targetfiles)

rename_entry_args = ('labels', 'labels')
bounds = np.s_[10:110, 200:712, 200:712]

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

        # Crop
        data.crop_bounding_rect(bounds)

        # Re-name entry
        data.rename_entry(*rename_entry_args)

        print 'data.dss(function=shp) = {}'.format(data.dss(function=shp))

        # Write to target
        data.write(filepath=targetfilepath)




