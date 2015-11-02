
import numpy
from volumina.api import Viewer
from PyQt4.QtGui import *
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from pyqtgraph.functions import arrayToQPath


class ClickablePathItem(QGraphicsPathItem):

    def __init__(self):
        super(ClickablePathItem, self).__init__()
        pen = QPen(QColor(255, 255, 0))
        pen.setWidth(1)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.TransparentMode))

    def mousePressEvent(self, event):

        if event.type() == QEvent.GraphicsSceneMousePress:

            if event.button() == QtCore.Qt.LeftButton:
                # print "MyPathItem left click event detected!"
                pen = QPen(QColor(0, 255, 255))
                pen.setWidth(5)
                self.setPen(pen)

            if event.button() == Qt.RightButton:
                # print "MyPathItem right click event detected!"
                pen = QPen(QColor(255, 255, 0))
                pen.setWidth(5)
                self.setPen(pen)

app = QtGui.QApplication([])

v = Viewer()

xFwd = numpy.array([10, 20, 40, 80])
xRev = xFwd[::-1]
x = numpy.concatenate([xFwd, xRev])
yFwd = numpy.array([80, 90, 100, 110])
yRev = yFwd[::-1]
y = numpy.concatenate([yFwd, yRev])
path1 = arrayToQPath(x, y)

item1 = ClickablePathItem()
item1.setPath(path1)

# Create images
back = (numpy.ones((100, 200, 300)) * 0).astype(numpy.uint8)
back[0:60, 0:60, 0:60] = 255
back[40:80, 40:80, 40:80] = 120
back[40:100, 40:100, 0:40] = 80
back[0:45, 50:100, 0:100] = 200
ol = (numpy.zeros((100, 200, 300))).astype(numpy.uint8)
# ol[0:99, 0:99, 0:99] = 120
# ol[0:99, 120:140, 0:99] = 255
ol[:] = back
back.shape = (1,)+back.shape+(1,)
ol.shape = (1,)+ol.shape+(1,)

# Add layers
l1 = v.addGrayscaleLayer(back, name="back")
l1.visible = True
l2 = v.addColorTableLayer(ol, name="overlay")


# Add the path to the graphics scene
# v.editor.imageScenes[0].addItem(item1)
# v.editor.imageScenes[1].addItem(item1)
v.editor.imageScenes[2].addItem(item1)

v.setWindowTitle("Clickable Border Viewer")
v.show()

app.exec_()
