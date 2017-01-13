
import numpy as np
import scipy
from scipy import ndimage, misc
import vigra
from copy import deepcopy
from vigra import graphs
import math

__author__ = 'jhennies'


def ellipsoid_se(shp, mode='outer', r_in=None):
    """
    Creates an ellipsoid structure element with given shape.
    :param shp: The shape
    :return:
    """
    if r_in is None: r_in = [0, 0, 0]

    shp = np.array(shp)

    if mode == 'center':
        r = [0, 0, 0]
    elif mode == 'outer':
        r = [0.5, 0.5, 0.5]
    elif mode == 'inner':
        r = [-0.5, -0.5, -0.5]
    elif mode == 'manual':
        r = r_in

    radii = (shp.astype(np.float) - 1) / 2

    def x(c):
        return c - radii[0]

    def y(c):
        return c - radii[1]

    def z(c):
        return c - radii[2]

    se = np.zeros(shp)

    for xi in xrange(0, int(shp[0])):
        for yi in xrange(0, int(shp[1])):
            for zi in xrange(0, int(shp[2])):

                se[xi, yi, zi] = (x(xi) / (radii[0] + r[0])) ** 2 \
                                 + (y(yi) / (radii[1] + r[1])) ** 2 \
                                 + (z(zi) / (radii[2] + r[2])) ** 2

    return se <= 1


if __name__ == '__main__':

    print ellipsoid_se((1, 5, 5), mode='manual', r_in=[0.5, 0, 0])