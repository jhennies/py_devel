
import numpy
from volumina.api import Viewer
# from PyQt4.QtGui import QPainterPath, QPen, QColor
from PyQt4.QtGui import *
from PyQt4 import QtGui, QtCore
# from PyQt4.QtCore import QEvent
from PyQt4.QtCore import *
# import h5py
# import sys
from pyqtgraph.functions import arrayToQPath


class MyPathItem(QGraphicsPathItem):

    state = 0

    def __init__(self):
        super(MyPathItem, self).__init__()
        self.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        # self.installSceneEventFilter(self)

    # def sceneEventFilter(self, item, event):
    #     print "Event in MyPathItem detected!"
    #     if item == self:
    #         if event.type() == QEvent.GraphicsSceneMousePress:
    #             if event.button() == QtCore.Qt.LeftButton:
    #                 print "MyPathItem left click event detected!"
    #                 return True
    #     return False

    def mousePressEvent(self, event):

        print "Event in MyPathItem detected!"
        if event.type() == QEvent.GraphicsSceneMousePress:
            if event.button() == QtCore.Qt.LeftButton:
                print "MyPathItem left click event detected!"
                if self.state == 0:
                    pen = QPen(QColor(0, 255, 255))
                    pen.setWidth(5)
                    self.setPen(pen)
                    self.state = 1
                else:
                    pen = QPen(QColor(255, 255, 0))
                    pen.setWidth(5)
                    self.setPen(pen)
                    self.state = 0

                # return True
        # return False

    # def shape(self):
    #     return self.path()


class ClickViewer(Viewer):

    def __init__(self):
        super(ClickViewer, self).__init__()
        self.editor.imageScenes[0].installEventFilter(self)
        # self.editor.imageScenes[1].installEventFilter(self)
        self.editor.imageScenes[2].installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj == self.editor.imageScenes[0]:
            if event.type() == QEvent.GraphicsSceneMousePress:
                if event.button() == QtCore.Qt.LeftButton:
                    print "ImageScenes[0] left click event detected!"
                    return True
        if obj == self.editor.imageScenes[1]:
            if event.type() == QEvent.GraphicsSceneMousePress:
                if event.button() == QtCore.Qt.LeftButton:
                    print "ImageScenes[1] left click event detected!"
                    return True
        if obj == self.editor.imageScenes[2]:
            if event.type() == QEvent.GraphicsSceneMousePress:
                if event.button() == QtCore.Qt.LeftButton:
                    print "ImageScenes[2] left click event detected!"
                    return True
        return False


app = QtGui.QApplication([])

v = ClickViewer()

print "MouseButtonPress: " + str(QEvent.MouseButtonPress)
print "LeftButton: " + str(QtCore.Qt.LeftButton)

# path1 = QPainterPath()
# path1.addRect(20, 20, 60, 60)
path2 = QPainterPath()
path2.moveTo(0, 0)
path2.cubicTo(99, 0,  50, 50,  99, 99)
path2.cubicTo(0, 99,  50, 50,  0, 0)

xFwd = numpy.array([10, 20, 40, 80])
xRev = xFwd[::-1]
x = numpy.concatenate([xFwd, xRev])
yFwd = numpy.array([80, 90, 100, 110])
yRev = yFwd[::-1]
y = numpy.concatenate([yFwd, yRev])
print x, y
path1 = arrayToQPath(x, y)
# path1 = QPainterPath()
# poly = QPolygonF()
# poly.append(QPointF(10, 80))
# poly.append(QPointF(20, 90))
# poly.append(QPointF(40, 100))
# poly.append(QPointF(80, 110))
# path1.addPolygon(poly)

item1 = MyPathItem()
item1.setPath(path1)
item1.setPen(QtGui.QPen(QColor(255, 255, 0)))
item1.setBrush(QBrush(Qt.TransparentMode))

# scene.addPath(path)

# view = QtGui.QGraphicsView(scene)
# view.show()

# Create images
back = (numpy.ones((100, 200, 300)) * 0).astype(numpy.uint8)
back[0:60, 0:60, 0:60] = 255
ol = (numpy.zeros((100, 200, 300))).astype(numpy.uint8)
ol[0:99, 0:99, 0:99] = 120
ol[0:99, 120:140, 0:99] = 255
back.shape = (1,)+back.shape+(1,)
ol.shape = (1,)+ol.shape+(1,)

# s = (ol/64).astype(numpy.uint8)

# Add a layer
l1 = v.addGrayscaleLayer(back, name="back", direct=True)
l1.visible = True

# v.editor.imageScenes is a 3D vector representing the QGraphicsScene for each of the three display planes
h_p1 = v.editor.imageScenes[2].addPath(path1, QPen(QColor(255, 255, 0)))
h_p2 = v.editor.imageScenes[2].addPath(path2, QPen(QColor(255, 0, 255)))
h_p2 = v.editor.imageScenes[0].addPath(path2, QPen(QColor(0, 255, 255)))

v.editor.imageScenes[1].addItem(item1)


# QtGui.QMouseEventTransition(v.editor.imageScenes[2], QtCore.QEvent.MouseButtonPress, QtCore.Qt.LeftButton, 0)

v.editor.imageScenes

v.setWindowTitle("Clickable Border Viewer")
v.show()

app.exec_()
