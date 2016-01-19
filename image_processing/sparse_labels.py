from make_membrane_label import ImageFileProcessing

__author__ = 'jhennies'

imagePath = '/windows/mobi/h1.hci/isbi_2013/data/xytc_boundaries.bl1.ign4.sparse/'
imageFile = 'xytc_boundaries.bl1.ign4.sparse1eq2.h5'
imageName = None
imageID = 0

# Initialize ImageFileProcessing object
ifp = ImageFileProcessing()
ifp.set_file(imagePath, imageFile, imageName, imageID)

num_lab2 = ifp.count_labels(label=2)
print num_lab2
num_lab1 = ifp.count_labels(label=1)
print num_lab1
n_lab1 = num_lab1 - 100000
n_lab2 = num_lab2 - 100000

ifp.randomly_convert_labels_h5(from_label=1, to_label=0, n=n_lab1,
                               file_name=None)
ifp.randomly_convert_labels_h5(from_label=2, to_label=0, n=n_lab2,
                               file_name='xytc_boundaries.bl1.ign4.sparse100k.h5')


# ifp.randomly_convert_labels_h5(from_label=1, to_label=0, n=1000,
#                                file_name=None)
# ifp.randomly_convert_labels_h5(from_label=2, to_label=0, n=1000,
#                                file_name='xytc_boundaries.blNone.ign4.ignl1.sparse1000.h5')
