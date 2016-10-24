
from image_processing import ImageFileProcessing

ifp = ImageFileProcessing(
    image_path='/media/julian/Daten/neuraldata/cremi_2016/',
    image_file='cremi.splA.raw_neurons.crop.crop_10-200-200_110-712-712.largeobjects.h5'
)

ifp.crop((16, 320, 180), (19, 350, 220), ids='largeobjects', targetids='crop')

ifp.filter_values(14994, type='eq', setto=0, ids='crop')

ifp.get_image('largeobjects')[16:19, 320:350, 180:220] = ifp.get_image('crop')

ifp.write(filename='largeobj_test.h5', ids=('background', 'labels', 'largeobjects'))