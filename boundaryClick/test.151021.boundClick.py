# import h5py
import numpy

from volumina.api import Viewer
# from PyQt4.QtGui import QApplication, QColor

from lazyflow.graph import Graph
# from lazyflow.operators.ioOperators.opStreamingHdf5Reader import OpStreamingHdf5Reader
# from lazyflow.operators import OpCompressedCache

from PyQt4 import QtCore, uic
from PyQt4.QtGui import QApplication, QPainter
from PyQt4 import QtGui
from PyQt4 import Qt

import pyqtgraph as pg
from pyqtgraph.functions import arrayToQPath


class rightLeftClickableViewer(Viewer):

    def __init__(self):
        Viewer.__init__(self)
        self.installEventFilter(self)

    def eventFilter(self, QObject, QEvent):
        if (QEvent.type() == QtCore.QEvent.MouseButtonRelease and
                QObject is self):
            pos = QEvent.pos()
            print('mouse button release: (%d, %d)' % (pos.x(), pos.y()))
        return Viewer.eventFilter(self, QObject, QEvent)

# from pyqtgraph.functions import arrayToQPath

# f = h5py.File("test.h5", 'w')
# data = (255 * numpy.ones((100, 200, 300))).astype(numpy.uint8)
# f.create_dataset("test", data=data)
# f.close()
#
# f = h5py.File("ol.h5", 'w')
# data = (255 * numpy.ones((100, 200, 300))).astype(numpy.uint8)
# f.create_dataset("ol", data=data)
# f.close()

# #-----

app = QApplication(sys.argv)
v = rightLeftClickableViewer()

graph = Graph()

# def mkH5source(fname, gname):
#     h5file = h5py.File(fname)
#     source = OpStreamingHdf5Reader(graph = graph)
#     source.Hdf5File.setValue(h5file)
#     source.InternalPath.setValue(gname)
#
#     op = OpCompressedCache(parent=None, graph=graph)
#     op.BlockShape.setValue([100, 100, 100])
#     op.Input.connect(source.OutputImage)
#
#     return op.Output
#
# testSource = mkH5source("test.h5", "test")
# olSource = mkH5source("ol.h5", "ol")

back = (numpy.ones((100, 200, 300)) * 0).astype(numpy.uint8)
back[0:60, 0:60, 0:60] = 255
ol = (numpy.zeros((100, 200, 300))).astype(numpy.uint8)
ol[0:99, 0:99, 0:99] = 120
ol[0:99, 120:140, 0:99] = 255

back.shape = (1,)+back.shape+(1,)
ol.shape = (1,)+ol.shape+(1,)

# s = (ol/64).astype(numpy.uint8)

l1 = v.addGrayscaleLayer(back, name="back", direct=True)
l1.visible = True

ol = ((ol/64)).astype(numpy.uint8)


def onClick(layer, pos5D, pos):
    print "Click: ", pos5D, ol[pos5D]


# def onLeftClick(layer, pos5D, pos):
#     print "leftClick: ", pos5D, ol[pos5D]


l2 = v.addColorTableLayer(ol, name="ol", direct=True, clickFunctor=onClick)
l2.visible = False



# v.addWidget(plot)


# l3 = v.addColorTableLayer(ol, name="paint", direct=True)



# v.addClickableSegmentationLayer(ol, "clickIt", direct=True)

# colortable = [QColor(0,0,0,0).rgba(), QColor(255,0,0).rgba(), QColor(0,255,0).rgba(), QColor(0,0,255).rgba()]
# l2 = v.addColorTableLayer(ol, name="ol", colortable=colortable, direct=True)
#
# # l3 = v.addColorTableLayer(s, name="s")
#
# # f = h5py.File("ol.h5", 'r')
# # raw = f["ol"].value
# # assert raw.ndim == 3
# # assert raw.dtype == numpy.uint8
# # f.close()
# #
# # raw.shape = (1,)+raw.shape+(1,)
# #
# # s = (raw/64).astype(numpy.uint8)
#
#
# def onClick(layer, pos5D, pos):
#     print "here i am: ", pos5D, s[pos5D]
#
# l2 = v.addColorTableLayer(ol, clickFunctor=onClick, name="thresh")
# l2.colortableIsRandom = True
# l2.zeroIsTransparent = True
# l2.visible = True
#
# # v.addClickableSegmentationLayer(s, "click it", direct=True)

x = numpy.array([200, 200, 200, 200])
y = numpy.array([10, 100, 200, 300])
qpp = arrayToQPath(x, y)
painter = QtGui.QPainter()
painter.begin(v)
painter.fillRect(0, 0, 100, 100, Qt.Qt.red)
painter.setPen(QtGui.QPen(QtGui.QColor(0, 100, 100), 1, Qt.Qt.SolidLine, Qt.Qt.FlatCap, Qt.Qt.MiterJoin))
painter.setBrush(QtGui.QColor(122, 163, 39))
painter.drawPath(qpp)
painter.end()

v.setWindowTitle("streaming viewer")
v.showMaximized()
app.exec_()



