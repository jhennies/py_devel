
from hdf5_image_processing import Hdf5ImageProcessing as IP, Hdf5ImageProcessingLib as IPL
import vigra
from vigra.learning import RandomForest as RF
from vigra.learning import RandomForestOld as RFO
# Sklearn
import numpy as np


__author__ = 'jhennies'


if __name__ == '__main__':

    # ipl = IPL()

    # ipl['fs'] = np.array([[0, 1, 0, 0, 0, 1, 2, 1, 0],
    #              [2, 2, 3, 2, 2, 3, 2, 1, 2],
    #              [6, 7, 4, 6, 5, 7, 5, 6, 7]], dtype=np.float64)
    #
    # ipl['fs'] = np.array([[0, 1, 0, 0, 0, 1, 2, 1, 0],
    #              [2, 2, 3, 2, 2, 3, 2, 1, 2],
    #              [6, 7, 4, 6, 5, 7, 5, 6, 7]], dtype=np.uint32)

    # ipl['ls'] = np.array([0, 1, 2], dtype=np.uint64)
    # ipl['fs'] = np.ones((10, 10, 3), dtype=np.float32)

    # rf = RF()
    # print rf.learnRF()
    # rf2 = RF()
    # # rf.learnRF(rf2, ipl['fs'], ipl['ls'])
    # rf.learnRF(rf2, 0.0, 1)

    # rf = RF(ipl['fs'], ipl['ls'])

    # trainData = ipl['fs']
    # trainLabels = ipl['ls']

    trainData = np.zeros((10, 3), dtype=np.float32)
    trainData[5:,0] = 1
    trainData[:5, 1] = 1
    print trainData
    trainLabels = np.zeros((10,), dtype=np.uint32)
    trainLabels[5:] = 1
    print trainLabels

    rf = RFO(trainData, trainLabels)#,
                # treeCount = 255, mtry=0, min_split_node_size=1,
                # training_set_size=0, training_set_proportions=1.0,
                # sample_with_replacement=True, sample_classes_individually=False,)
    print rf



    # rf = RF()
    # rf2 = RF()
    # rf.learnRF(trainData, trainLabels)
