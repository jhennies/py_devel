
import numpy
from volumina.api import Viewer
from PyQt4.QtGui import *
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from pyqtgraph.functions import arrayToQPath
from volumina.layer import GrayscaleLayer
from volumina.layer import ColortableLayer
import vigra
import _image_processing


class ClickablePathItem(QGraphicsPathItem):

    _currentPen = QPen()

    def __init__(self):
        super(ClickablePathItem, self).__init__()
        self.setAcceptTouchEvents(True)
        self.setAcceptHoverEvents(True)

    def set_path_from_coordinates(self, x, y):
        # TODO: Going back along the path to avoid creation of an area by the automatically closing path may not be the
        # best solution...
        x_fwd = x
        x_rev = x_fwd[::-1]
        x = numpy.concatenate([x_fwd, x_rev])
        y_fwd = y
        y_rev = y_fwd[::-1]
        y = numpy.concatenate([y_fwd, y_rev])
        path = arrayToQPath(x, y)
        self.setPath(path)

    def setPath(self, painter_path):
        super(ClickablePathItem, self).setPath(painter_path)
        # Set invisible wide path which throws click events

    def mousePressEvent(self, event):

        if event.type() == QEvent.GraphicsSceneMousePress:

            if event.button() == QtCore.Qt.LeftButton:
                # print "MyPathItem left click event detected!"
                pen = QPen(QColor(0, 255, 255))
                self._currentPen = pen
                pen.setWidth(5)
                self.setPen(pen)

            if event.button() == Qt.RightButton:
                # print "MyPathItem right click event detected!"
                pen = QPen(QColor(255, 255, 0))
                self._currentPen = pen
                pen.setWidth(5)
                self.setPen(pen)

    def hoverEnterEvent(self, event):
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(5)
        self.setPen(pen)

    def hoverLeaveEvent(self, event):
        pen = self._currentPen
        pen.setWidth(0)
        self.setPen(pen)


class BoundaryViewer(Viewer):

    _back_image = numpy.array
    _overlay_image = numpy.array
    _layer_back = GrayscaleLayer
    _layer_overlay = ColortableLayer

    _visible_boundaries = []
    _visible_paths = [[]]

    _clickable_paths = []

    def __init__(self):
        super(BoundaryViewer, self).__init__()
        self.editor.posModel.slicingPositionChanged.connect(self.slicing_position_changed)

    def add_background_image(self, image):
        self._back_image = image
        self._layer_back = self.addGrayscaleLayer(image, name="back")
        self._layer_back.visible = True
        return self._layer_back

    def add_overlay(self, overlay):
        self._overlay_image = overlay
        self._layer_overlay = self.addColorTableLayer(overlay, name="overlay")
        self._layer_overlay.visible = True
        return self._layer_overlay

    @pyqtSlot(object, object)
    def slicing_position_changed(self, new_position, old_position):
        # print "Slicing position changed!"
        # print new_position
        # print old_position
        path_data = self.get_visible_paths(new_position)
        # print path_data
        # print len(path_data)
        # print path_data[0]
        # print len(path_data[0])
        # print path_data[1]
        # print len(path_data[1])

        for o in self._clickable_paths:
            self.editor.imageScenes[2].removeItem(o)

        self._clickable_paths = []

        for pd in path_data[1]:
            # c_x = numpy.array([10, 20, 40, 80]) + 20 * i
            # c_y = numpy.array([80, 90, 100, 110])
            c_x = pd[0]
            c_y = pd[1]
            print c_x
            print c_y
            # path = arrayToQPath(c_x, c_y)
            self._clickable_paths.append(ClickablePathItem())
            self._clickable_paths[-1].set_path_from_coordinates(c_x, c_y)
            # self._clickable_paths[-1].setPath(path)
            # self._clickable_paths[-1].add_item_to_scene()
            pen = QPen(QColor(255, 255, 255))
            pen.setWidth(0)
            self._clickable_paths[-1].setPen(pen)

            self.editor.imageScenes[2].addItem(self._clickable_paths[-1])

    def get_visible_paths(self, position):
        current_slice_z = numpy.array(self._overlay_image[0, :, :, position[2], 0], dtype=numpy.float32)
        return _image_processing.findEdgePaths(current_slice_z)

    def determine_edge_pixels(self, label1, label2):
        return 0

########################################################################################################################
# __main__
########################################################################################################################
if __name__ == "__main__":

    app = QtGui.QApplication([])

    v = BoundaryViewer()

    # c_x = numpy.array([10, 20, 40, 80])
    # c_y = numpy.array([80, 90, 100, 110])
    # item1 = ClickablePathItem(v.editor.imageScenes[2], clickable_width=11)
    # item1.set_path_from_coordinates(c_x, c_y)
    # item2 = ClickablePathItem(v.editor.imageScenes[2], clickable_width=11)
    # item2.set_path_from_coordinates(c_x, c_y)
    # item1.add_item_to_scene()
    # item2.add_item_to_scene()


    # v.editor.imageScenes[0]._onSlicingPositionChanged()

    # Create images
    back = (numpy.ones((100, 200, 300)) * 0).astype(numpy.uint8)
    back[0:60, 0:60, 0:60] = 255
    back[40:80, 40:80, 40:80] = 120
    back[40:100, 40:100, 0:40] = 80
    back[0:45, 50:100, 0:100] = 200
    back[back == 0] = 1

    ol = (numpy.zeros((100, 200, 300))).astype(numpy.uint8)
    # ol[0:99, 0:99, 0:99] = 120
    # ol[0:99, 120:140, 0:99] = 255
    ol[:] = back
    back.shape = (1,)+back.shape+(1,)
    ol.shape = (1,)+ol.shape+(1,)

    # Add layers
    # l1 = v.addGrayscaleLayer(back, name="back")
    # l1.visible = True
    # l2 = v.addColorTableLayer(ol, name="overlay")
    l1 = v.add_background_image(back)
    l2 = v.add_overlay(ol)

    # Add the path to the graphics scene
    # v.editor.imageScenes[0].addItem(item1)
    # v.editor.imageScenes[1].addItem(item1)
    # v.editor.imageScenes[2].addItem(item1)

    v.setWindowTitle("Clickable Border Viewer")
    v.show()

    app.exec_()

