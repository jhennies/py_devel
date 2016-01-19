from make_membrane_label import ImageFileProcessing

__author__ = 'jhennies'

imagePath = '/windows/mobi/h1.hci/isbi_2013/prediction/20160105.blNone.ign4.ignl1.sparse10k/'
imageFile = 'xytc_test-data-probs_probs.h5'
imageName = None
imageID = 0

# Initialize ImageFileProcessing object
ifp = ImageFileProcessing()
ifp.set_file(imagePath, imageFile, imageName, imageID)

# Split data
ifp.split_data_h5(file_name='split_data.h5', split_dimension=4, squeeze=True)
