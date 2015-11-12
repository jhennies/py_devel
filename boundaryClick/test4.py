# import _my_module

import numpy
import _image_processing

__author__ = 'jhennies'

img = numpy.random.rand(4, 3)
ret = _image_processing.imageLoop(numpy.require(img, dtype=numpy.float32))

print img, '\n', ret

# _my_module.imageLoop()
print _image_processing.testFunction(1)
# _my_module.export_HelloWorld()

labels = numpy.zeros((5, 4), dtype=numpy.float32)
labels[:] = 3
labels[:, 0] = 1
labels[0:2, 1] = 1
labels[0:2, 2:4] = 2
labels[2, 1:3] = 2

# labels = numpy.zeros((10, 10), dtype=numpy.float32)
# labels[:] = 1
# labels[6:10, :] = 2
# labels[5, 0:3] = 2
# labels[5, 7:10] = 2

print labels
print _image_processing.findEdgePaths(labels)


class TestClass:

    _a = 0

    def __init__(self, a):
        self._a = a


tc = TestClass(1)

print tc._a

tc2 = TestClass(1)
tc2._a = 2

print tc._a
print tc2._a