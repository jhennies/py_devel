
# import h5py
import numpy
import sys

from volumina.api import Viewer
# from PyQt4.QtGui import QApplication, QColor

from lazyflow.graph import Graph
# from lazyflow.operators.ioOperators.opStreamingHdf5Reader import OpStreamingHdf5Reader
# from lazyflow.operators import OpCompressedCache

from PyQt4 import QtCore, uic
from PyQt4.QtGui import QApplication, QPainter, QPainterPath
from PyQt4 import QtGui
from PyQt4 import Qt

app = QtGui.QApplication([])

scene = QtGui.QGraphicsScene()
scene.addText("Hello World!")

back = (numpy.ones((100, 200, 300)) * 0).astype(numpy.uint8)
back[0:60, 0:60, 0:60] = 255
ol = (numpy.zeros((100, 200, 300))).astype(numpy.uint8)
ol[0:99, 0:99, 0:99] = 120
ol[0:99, 120:140, 0:99] = 255

path = QPainterPath()
path.addRect(20, 20, 60, 60)
path.moveTo(0, 0)
path.cubicTo(99, 0,  50, 50,  99, 99)
path.cubicTo(0, 99,  50, 50,  0, 0)

scene.addPath(path)

view = QtGui.QGraphicsView(scene)
view.show()

app.exec_()
