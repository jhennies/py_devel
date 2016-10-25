
from hdf5_image_processing import Hdf5ImageProcessingLib as IPL
import random
import vigra.graphs as graphs
import numpy as np
import os
import inspect
from shutil import copy
import processing_lib as lib
from copy import deepcopy
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
import vigra
from skimage import morphology

__author__ = 'jhennies'


def boundary_disttransf(hfp, thisparams):
    # Boundary distance transform
    # a) Boundaries
    hfp.logging('Finding boundaries ...')
    hfp.pixels_at_boundary(
        axes=(np.array(thisparams['anisotropy']).astype(np.float32) ** -1).astype(np.uint8),
        tkeys='disttransf'
    )
    hfp.astype(np.float32)

    # b) Distance transform
    hfp.logging('Computing distance transform on boundaries ...')
    hfp.distance_transform(
        pixel_pitch=thisparams['anisotropy'],
        background=True,
        keys='disttransf'
    )


def compute_faces(hfp):

    hfp.logging('Comuting faces ...')
    hfp.get_faces_with_neighbors(keys=('largeobj', 'disttransf'), tkeys='faces')
    hfp.logging('hfp.datastructure\n---\n{}', hfp.datastructure2string())


def find_border_contacts(hfp):

    params = hfp.get_params()
    thisparams = params['find_border_contacts']

    # # Clear objects at the border
    # hfp.anytask(clear_border)

    # Do it manually
    # --------------
    # Compute the distance transform
    boundary_disttransf(hfp, thisparams)

    # For each of the 6 faces compute the objects which are touching it and the corresponding local maxima of the
    # distance transform
    compute_faces(hfp)

    # Find global maxima for each object touching the border
    shp = hfp['largeobj'].shape
    hfp['border_locmax'] = np.zeros(shp)

    keys = {'xyf': shp[2],
            'xyb': shp[2],
            'xzf': shp[1],
            'xzb': shp[1],
            'yzf': shp[0],
            'yzb': shp[0]}
    for k, bounds in keys.iteritems():

        # bounds = (shp[0],) * 2
        for lbl, lblim in hfp['faces', 'largeobj'].label_image_iterator(key=k, background=0):

            hfp.logging('---\nLabel {} found in image {}', lbl, k)

            # Avoid very small artifacts
            lblim = morphology.opening(lblim)

            # Connected component analysis to detect when a label touches the border multiple times
            conncomp = vigra.analysis.labelImageWithBackground(lblim.astype(np.uint32), neighborhood=8, background_value=0)

            for l in np.unique(conncomp):
                if l == 0: continue
                # print l
                # plt.figure()
                curobj = conncomp == l
                # print np.amax(curobj)
                # plt.imshow(conncomp==l)
                # plt.show()

                # Get disttancetransf of the object
                curdist = np.array(hfp['faces', 'disttransf', k])
                curdist[curobj == False] = 0
                # plt.imshow(curdist)
                # plt.show()

                # Detect the global maximum of this object
                # Only one pixel is allowed to be selected
                amax = np.amax(curdist)
                curdist[curdist < amax] = 0
                curdist[curdist > 0] = lbl
                bds = lib.find_bounding_rect(curdist)
                centroid = (int((bds[1][0] + bds[1][1]-1) / 2), int((bds[0][0] + bds[0][1]-1) / 2))

                # TODO: bounds depends on k
                centroidm = (centroid[0] - bounds, centroid[1] - bounds)
                hfp.logging('centroidxy = {}', centroidm)
                # Set the pixel
                try:
                    if centroidm[0] < 0 or centroidm[1] < 0:
                        raise IndexError
                    else:
                        if k == 'xyf':
                            hfp['border_locmax'][centroidm[0], centroidm[1], 0] = lbl
                        elif k == 'xyb':
                            hfp['border_locmax'][centroidm[0], centroidm[1], -1] = lbl
                        elif k == 'xzf':
                            hfp['border_locmax'][centroidm[0], 0, centroidm[1]] = lbl
                        elif k == 'xzb':
                            hfp['border_locmax'][centroidm[0], -1, centroidm[1]] = lbl
                        elif k == 'yzf':
                            hfp['border_locmax'][0, centroidm[0], centroidm[1]] = lbl
                        elif k == 'yzb':
                            hfp['border_locmax'][-1, centroidm[0], centroidm[1]] = lbl
                except IndexError:
                    pass

    # # Create data structure for output
    # hfp.rename_entry('largeobj', 'orphans')

    hfp['overlay'] = np.array([(hfp['border_locmax'] > 0).astype(np.float32), hfp['largeobj']/np.amax(hfp['largeobj']), hfp['disttransf']/np.amax(hfp['disttransf'])])
    # ifp.set_data_dict({'paths_over_dist': np.array([ifp.get_image('pathsim'), ifp.get_image('curlabelpair'), ifp.get_image('curdisttransf')])}, append=True)

if __name__ == '__main__':

    yamlfile = os.path.dirname(os.path.abspath(__file__)) + '/parameters.yml'

    hfp = IPL(
        yaml=yamlfile,
        yamlspec={'path': 'datafolder', 'filename': 'largeobjfile', 'skeys': 'largeobjname'},
        tkeys='largeobj',
        castkey=None
    )
    params = hfp.get_params()
    hfp.startlogger(filename=params['resultfolder'] + 'find_orphans.log', type='w')

    try:

        # Copy the script file and the parameters to the scriptsfolder
        copy(inspect.stack()[0][1], params['scriptsfolder'])
        copy(yamlfile, params['scriptsfolder'] + 'find_orphans.parameters.yml')
        # Write script and parameters to the logfile
        hfp.code2log(inspect.stack()[0][1])
        hfp.logging('')
        hfp.yaml2log()
        hfp.logging('')

        hfp.logging('\nhfp datastructure: \n---\n{}', hfp.datastructure2string(maxdepth=1))

        find_border_contacts(hfp)

        # TODO: Comment in when ready
        hfp.write(filepath=params['intermedfolder'] + params['orphansfile'])

        hfp.logging('\nFinal dictionary structure:\n---\n{}', hfp.datastructure2string())
        hfp.logging('')
        hfp.stoplogger()

    except:

        hfp.errout('Unexpected error')