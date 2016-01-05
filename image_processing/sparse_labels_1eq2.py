from make_membrane_label import ImageFileProcessing

__author__ = 'jhennies'

imagePath = '/windows/mobi/h1.hci/isbi_2013/data/'
imageFile = 'xytc_boundaries.blNone.ign4.ignl1.h5'
imageName = None
imageID = 0

# Initialize ImageFileProcessing object
ifp = ImageFileProcessing()
ifp.set_file(imagePath, imageFile, imageName, imageID)

num_lab2 = ifp.count_labels(label=2)
print num_lab2
num_lab1 = ifp.count_labels(label=1)
print num_lab1
n = num_lab1 - num_lab2
if n > 0:
    ifp.randomly_convert_labels_h5(from_label=1, to_label=0, n=n,
                                   file_name='xytc_boundaries.blNone.ign4.ignl1.sparse.h5')
else:
    print 'n <= 0'