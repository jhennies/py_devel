from make_membrane_label import ImageFileProcessing

__author__ = 'jhennies'

imagePath = '/windows/mobi/h1.hci/isbi_2013/prediction/2015.test.crop_100_100_100/'
imageFile = 'xytc_test-data-probs.crop_100_100_100_probs.h5'
imageName = None
imageID = 0

# Initialize ImageFileProcessing object
ifp = ImageFileProcessing()
ifp.set_file(imagePath, imageFile, imageName, imageID)

# Split data
ifp.split_data_h5(file_name='split_data.h5', split_dimension=4, squeeze=True)
