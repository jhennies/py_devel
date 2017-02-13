
from hdf5_image_processing import Hdf5ImageProcessingLib as ipl
import processing_lib as lib
import numpy as np

debug = False

path = '/mnt/localdata01/jhennies/neuraldata/cremi_2016/sample_BC_train_betas/'

# Set sourcenames
sourcename = 'cremi.splC.train.mcseg_{}.crop.split_xyz.h5'
ids = ['beta_0.3', 'beta_0.35', 'beta_0.4', 'beta_0.45', 'beta_0.5', 'beta_0.55', 'beta_0.6', 'beta_0.65', 'beta_0.7']

# Set targetname
targetname = 'cremi.splC.train.mcseg_betas.crop.split_xyz.h5'

# Generate file name lists
# sourcefiles = [sourcename.format(x) for x in ids]
# print 'files = {}'.format(sourcefiles)
targetfile = targetname
print 'targetfile = {}'.format(targetfile)

# Parameters
skey = 'labels'

for id in ids:

    source = sourcename.format(id)
    sourcefilepath = path + source
    print 'sourcefilepath = {}'.format(sourcefilepath)

    targetfilepath = path + targetfile

    if not debug:

        # Load data
        data = ipl()
        data.data_from_file(filepath=sourcefilepath, skeys=skey, recursive_search=True, nodata=True)
        data.rename_entry(skey, id, search=True)

        # Print shape
        def shp(x):
            return x.shape
        print 'data.dss(function=shp) = {}'.format(data.dss(function=shp))

        # Write to target
        data.write(filepath=targetfilepath)




