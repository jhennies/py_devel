from make_membrane_label import ImageFileProcessing

__author__ = 'jhennies'

imagePath = '/windows/mobi/h1.hci/isbi_2013/data/'
imageFile = 'test-input.h5'
imageName = None
imageID = 0

# Initialize ImageFileProcessing object
ifp = ImageFileProcessing()
ifp.set_file(imagePath, imageFile, imageName, imageID)

# # Crop image if necessary, then re-run with cropped image
# ifp.crop_h5((100, 100, 100))
# quit()

ifp.stack_h5('test-data-probs.h5',
             '/windows/mobi/h1.hci/isbi_2013/data/test-probs-nn-m.h5', stack_id=0)
