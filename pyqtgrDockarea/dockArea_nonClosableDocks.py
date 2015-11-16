from pyqtgraph.dockarea import *
from PyQt4 import QtGui, QtCore
from pyqtgraph.dockarea.DockArea import *

from PyQt4 import Qt
# from PyQt4.QtGui import *


__author__ = 'jhennies'


class MyDockArea(DockArea):
    def __init__(self, *args,**kwargs):
        super(MyDockArea, self).__init__(*args, **kwargs)

    def clear(self):
        docks = self.findAll()[1]
        for dock in docks.values():
            print "CLOSE DOCK"
            if dock.closable:
                dock.close()
            else:
                self.home.moveDock(dock, "top", None)

    def addTempArea(self):
        if self.home is None:
            area = DockArea(temporary=True, home=self)
            self.tempAreas.append(area)
            win = TempAreaWindow(area)
            win.setWindowFlags(Qt.Qt.WindowMinimizeButtonHint)
            area.win = win
            win.show()
        else:
            area = self.home.addTempArea()
        #print "added temp area", area, area.window()
        return area


########################################################################################################################
# __main__
########################################################################################################################
if __name__ == "__main__":

    app = QtGui.QApplication([])

    mw = QtGui.QMainWindow()
    mw.setWindowTitle('nonClosableDocks')
    area = MyDockArea()
    mw.setCentralWidget(area)
    mw.resize(800, 600)

    d1 = Dock("Dock1", size=(1, 1), closable=False)     ## give this dock the minimum possible size
    d2 = Dock("Dock2 - Console", size=(500,300), closable=True)
    d3 = Dock("Dock3", size=(500,400))
    d4 = Dock("Dock4 (tabbed) - Plot", size=(500,200))
    d5 = Dock("Dock5 - Image", size=(500,200))
    d6 = Dock("Dock6 (tabbed) - Plot", size=(500,200))

    # Modify docks #####################################################################################################
    # d2.setWindowFlags(Qt.Qt.CustomizeWindowHint)
    # mw.setWindowFlags(Qt.Qt.WindowCloseButtonHint)

    area.addDock(d1, 'left')      ## place d1 at left edge of dock area (it will fill the whole space since there are no other docks yet)
    area.addDock(d2, 'right')     ## place d2 at right edge of dock area
    area.addDock(d3, 'bottom', d1)## place d3 at bottom edge of d1
    area.addDock(d4, 'right')     ## place d4 at right edge of dock area
    area.addDock(d5, 'left', d1)  ## place d5 at left edge of d1
    area.addDock(d6, 'top', d4)   ## place d5 at top edge of d4

    mw.show()

    app.exec_()

