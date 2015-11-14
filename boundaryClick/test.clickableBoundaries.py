
import numpy
from volumina.api import Viewer
from volumina.pixelpipeline.datasources import *
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
        x[:] = [i + 0.5 for i in x]
        y[:] = [i + 0.5 for i in y]
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

    def setPen(self, q_pen):
        super(ClickablePathItem, self).setPen(q_pen)
        self._currentPen = q_pen

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
        # pen = QPen(QColor(255, 0, 0))
        pen = self._currentPen
        pen.setWidth(5)
        self.setPen(pen)

    def hoverLeaveEvent(self, event):
        pen = self._currentPen
        pen.setWidth(0)
        self.setPen(pen)


class BoundaryViewer(Viewer):

    # _back_image = numpy.array
    # _overlay_image = numpy.array
    _layer_back = GrayscaleLayer
    # _layer_overlay = ColortableLayer

    _clickable_paths = [[]]
    _overlay_layer = ColortableLayer

    def __init__(self):
        super(BoundaryViewer, self).__init__()
        self.editor.posModel.slicingPositionChanged.connect(self.slicing_position_changed)

    def add_layer_from_datasource(self, source, name=None, colortable=None, direct=False):
        if colortable is None:
            colortable = self._randomColors()
        self._overlay_layer = ColortableLayer(source, colortable, direct=direct)
        if name:
            self._overlay_layer.name = name
        self.layerstack.append(self._overlay_layer)
        return self._overlay_layer

    def add_background_image(self, image):
        # self._back_image = image
        self._layer_back = self.addGrayscaleLayer(image, name="back")
        self._layer_back.visible = True
        return self._layer_back

    # def add_overlay(self, overlay):
    #     self._overlay_image = overlay
    #     self._layer_overlay = self.addColorTableLayer(overlay, name="overlay")
    #     self._layer_overlay.visible = True
    #     return self._layer_overlay

    @pyqtSlot(object, object)
    def slicing_position_changed(self, new_position, old_position):

        # TODO: This code has to be also executed when the viewport rect changes

        # Calculate paths for each of the three displays
        for i in range(0, 3):

            if self._clickable_paths:
                for p in self._clickable_paths[i]:
                    self.editor.imageScenes[i].removeItem(p)
                else:
                    self._clickable_paths.append([])

            self._clickable_paths[i] = []

            path_data = self.get_visible_paths(new_position, i)

            for pd in path_data[1]:
                c_x = pd[0]
                c_y = pd[1]
                self._clickable_paths[i].append(ClickablePathItem())
                self._clickable_paths[i][-1].set_path_from_coordinates(c_x, c_y)
                pen = QPen(QColor(255, 0, 0))
                pen.setWidth(0)
                self._clickable_paths[i][-1].setPen(pen)

                self.editor.imageScenes[i].addItem(self._clickable_paths[i][-1])

    def get_visible_paths(self, position, display_plane):
        # TODO: Here we need to obtain the image using the data source object
        # if display_plane == 0:
        #     current_slice = numpy.transpose(numpy.array(self._overlay_image[0, position[0], :, :, 0], dtype=numpy.float32))
        # elif display_plane == 1:
        #     current_slice = numpy.array(self._overlay_image[0, :, position[1], :, 0], dtype=numpy.float32)
        # elif display_plane == 2:
        #     current_slice = numpy.array(self._overlay_image[0, :, :, position[2], 0], dtype=numpy.float32)
        #
        # return _image_processing.findEdgePaths(current_slice)

        # TODO: replace this by determination of the actual bounding box
        rq = self._overlay_layer.datasources[0].request(
            (slice(0, 1),
             slice(10, 70),
             slice(10, 70),
             slice(10, 70),
             slice(0, 1)
             ))

        print self.editor.imageViews[0].viewportRect()

        # Get request data and convert it into an array
        slc = numpy.asarray(rq.wait(), dtype=numpy.float32)
        # print slc.shape
        return _image_processing.findEdgePaths(slc[0, :, :, position[2], 0])

        # # TODO: This is just a dummy to avoid errors
        # return _image_processing.findEdgePaths(numpy.zeros((10, 10), dtype=numpy.float32))

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
    # back = (numpy.ones((100, 200, 300)) * 0).astype(numpy.uint8)
    # back[:] = 2
    # back[0:60, 0:60, 0:60] = 1
    # ol = (numpy.zeros((100, 200, 300))).astype(numpy.uint8)
    # ol[:] = back

    back = (numpy.ones((100, 200, 300)) * 0).astype(numpy.uint8)
    back[:] = 1
    back[0:60, 0:60, 0:60] = 255
    back[40:80, 40:80, 40:80] = 120
    back[40:100, 40:100, 0:40] = 80
    back[0:45, 50:100, 0:100] = 200
    ol = (numpy.zeros((100, 200, 300))).astype(numpy.uint8)
    # ol[0:99, 0:99, 0:99] = 120
    # ol[0:99, 120:140, 0:99] = 255
    ol[:] = back

    # back = (numpy.ones((20, 20, 20)) * 0).astype(numpy.uint8)
    # back[:] = 2
    # back[0:5, 0:5, 0:5] = 1
    # ol = (numpy.zeros((20, 20, 20))).astype(numpy.uint8)
    # ol[:] = back

    # back = (numpy.ones((500, 500, 500)) * 0).astype(numpy.uint16)
    # back[:] = 1
    # for i in range(0, 500, 50):
    #     back[i:i+50, :, :] = back[i:i+50, :, :] + i
    #     back[:, i:i+50, :] = back[:, i:i+50, :] + i*10
    #     back[:, :, i:i+50] = back[:, :, i:i+50] + i*100
    # ol = (numpy.zeros((500, 500, 500))).astype(numpy.uint16)
    # ol[:] = back

    back.shape = (1,)+back.shape+(1,)
    ol.shape = (1,)+ol.shape+(1,)

    # Add layers
    l1 = v.add_background_image(back)
    # l2 = v.add_overlay(ol)
    src = ArraySource(ol)
    l2 = v.add_layer_from_datasource(src, name="ol", colortable=None, direct=False)

    v.setWindowTitle("Clickable Border Viewer")
    v.show()

    app.exec_()

