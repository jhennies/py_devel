
from hdf5_image_processing import Hdf5ImageProcessingLib as IPL


__author__ = 'jhennies'


if __name__ == "__main__":

    # cremi = IPL(filepath='/mnt/localdata02/jhennies/neuraldata/cremi_2016/sample_B_20160501.hdf')
    #
    # cremi.logging('Datastructure:\n---\n{}', cremi.datastructure2string())
    #
    # images = IPL(data={
    #     'raw': cremi['volumes', 'raw'],
    #     'neuron_ids': cremi['volumes', 'labels', 'neuron_ids']
    # })
    #
    # images.logging('Datastructure:\n---\n{}', images.datastructure2string())
    #
    # images.write('/mnt/localdata02/jhennies/neuraldata/cremi_2016/cremi.splB.raw_neurons.crop.h5')



    cremi = IPL(filepath='/mnt/localdata02/jhennies/neuraldata/cremi_2016/sample_C_20160501.hdf')

    cremi.logging('Datastructure:\n---\n{}', cremi.datastructure2string())

    images = IPL(data={
        'raw': cremi['volumes', 'raw'],
        'neuron_ids': cremi['volumes', 'labels', 'neuron_ids']
    })

    images.logging('Datastructure:\n---\n{}', images.datastructure2string())

    images.write('/mnt/localdata02/jhennies/neuraldata/cremi_2016/cremi.splC.raw_neurons.crop.h5')