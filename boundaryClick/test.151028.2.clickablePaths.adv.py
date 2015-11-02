
import numpy
from volumina.api import Viewer
from PyQt4.QtGui import *
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from pyqtgraph.functions import arrayToQPath


class ClickablePathItem(QGraphicsPathItem):

    visiblePath = QGraphicsPathItem()
    graphicsScene = QGraphicsScene
    clickableWidth = 5

    def __init__(self, graphics_scene, clickable_width=5):
        super(ClickablePathItem, self).__init__()
        self.graphicsScene = graphics_scene
        self.graphicsScene.addItem(self)
        self.graphicsScene.addItem(self.visiblePath)
        self.clickableWidth = clickable_width

    def set_path_from_coordinates(self, x, y):
        x_fwd = x
        x_rev = x_fwd[::-1]
        x = numpy.concatenate([x_fwd, x_rev])
        y_fwd = y
        y_rev = y_fwd[::-1]
        y = numpy.concatenate([y_fwd, y_rev])
        path = arrayToQPath(x, y)
        self.setPath(path)

    def setPath(self, painter_path):
        # Set invisible wide path which throws click events
        super(ClickablePathItem, self).setPath(painter_path)
        pen = QPen(QColor(0, 0, 0, 0))
        pen.setWidth(self.clickableWidth)
        self.setPen(pen)

        # Set visible narrow path for display
        self.visiblePath.setPath(painter_path)
        pen = QPen(QColor(255, 255, 0))
        pen.setWidth(1)
        self.visiblePath.setPen(pen)

    def set_clickable_width(self, clickable_width):
        self.clickableWidth = clickable_width
        pen = QPen(QColor(0, 0, 0, 0))
        pen.setWidth(self.clickableWidth)
        self.setPen(pen)

    def mousePressEvent(self, event):

        if event.type() == QEvent.GraphicsSceneMousePress:

            if event.button() == QtCore.Qt.LeftButton:
                # print "MyPathItem left click event detected!"
                pen = QPen(QColor(0, 255, 255))
                self.visiblePath.setPen(pen)

            if event.button() == Qt.RightButton:
                # print "MyPathItem right click event detected!"
                pen = QPen(QColor(255, 255, 0))
                self.visiblePath.setPen(pen)


app = QtGui.QApplication([])

v = Viewer()

c_x = numpy.array([10, 20, 40, 80])
c_y = numpy.array([80, 90, 100, 110])
item1 = ClickablePathItem(v.editor.imageScenes[2], clickable_width=11)
item1.set_path_from_coordinates(c_x, c_y)

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
# v.editor.imageScenes[2].addItem(item1)

v.setWindowTitle("Clickable Border Viewer")
v.show()

app.exec_()
