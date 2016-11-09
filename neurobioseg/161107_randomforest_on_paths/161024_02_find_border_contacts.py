
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


def boundary_disttransf(ipl, thisparams, key, tkey):

    ipl.logging('Computing boundary distance transform for key = {}', key)
    # Boundary distance transform
    # a) Boundaries
    ipl.logging('Finding boundaries ...')
    ipl.pixels_at_boundary(
        axes=(np.array(thisparams['anisotropy']).astype(np.float32) ** -1).astype(np.uint8),
        keys=key,
        tkeys=tkey
    )
    ipl.astype(np.float32, keys=tkey)

    # b) Distance transform
    ipl.logging('Computing distance transform on boundaries ...')
    ipl.distance_transform(
        pixel_pitch=thisparams['anisotropy'],
        background=True,
        keys=tkey
    )


def compute_faces(ipl, keys, tkeys):

    ipl.logging('Computing faces ...')
    ipl.get_faces_with_neighbors(keys=keys, tkeys=tkeys)
    ipl.logging('ipl.datastructure\n---\n{}', ipl.datastructure2string())


def find_border_centroids(ipl, keys, areas, largeobjkey, disttransfkey, resultkey):

    for k, bounds in keys.iteritems():

        # bounds = (shp[0],) * 2
        for lbl, lblim in ipl['faces', largeobjkey].label_image_iterator(key=k, background=0, area=areas[k]):

            ipl.logging('---\nLabel {} found in image {}', lbl, k)

            # Avoid very small artifacts
            lblim = morphology.opening(lblim)

            # Connected component analysis to detect when a label touches the border multiple times
            conncomp = vigra.analysis.labelImageWithBackground(lblim.astype(np.uint32), neighborhood=8, background_value=0)

            for l in np.unique(conncomp):
                # Ignore background
                if l == 0: continue

                # Get the current label object
                curobj = conncomp == l

                # Get disttancetransf of the object
                curdist = np.array(ipl['faces', disttransfkey, k])
                curdist[curobj == False] = 0

                # Detect the global maximum of this object
                amax = np.amax(curdist)
                curdist[curdist < amax] = 0
                curdist[curdist > 0] = lbl
                # Only one pixel is allowed to be selected
                try:
                    bds = lib.find_bounding_rect(curdist)
                except ValueError:
                    # A value error is thrown when the current object is just one pixel in size
                    # This can be ignored without ignoring relevant border contacts
                    pass

                centroid = (int((bds[1][0] + bds[1][1]-1) / 2), int((bds[0][0] + bds[0][1]-1) / 2))

                # Now translate the calculated centroid to the position within the orignial 3D volume
                centroidm = (centroid[0] - bounds, centroid[1] - bounds)
                # ipl.logging('centroidxy = {}', centroidm)
                # Set the pixel
                try:
                    if centroidm[0] < 0 or centroidm[1] < 0:
                        raise IndexError
                    else:
                        if k == 'xyf':
                            ipl[resultkey][centroidm[0], centroidm[1], 0] = lbl
                        elif k == 'xyb':
                            ipl[resultkey][centroidm[0], centroidm[1], -1] = lbl
                        elif k == 'xzf':
                            ipl[resultkey][centroidm[0], 0, centroidm[1]] = lbl
                        elif k == 'xzb':
                            ipl[resultkey][centroidm[0], -1, centroidm[1]] = lbl
                        elif k == 'yzf':
                            ipl[resultkey][0, centroidm[0], centroidm[1]] = lbl
                        elif k == 'yzb':
                            ipl[resultkey][-1, centroidm[0], centroidm[1]] = lbl
                except IndexError:
                    pass


def find_orphans(ipl, bordercontacts, key, tkey):

    non_orphan_labels = []
    for k, v in ipl['faces', key].iteritems():
        non_orphan_labels = np.unique(np.append(v, non_orphan_labels))

    all_labels = np.unique(ipl[key])
    orphan_labels = list(set(all_labels).difference(non_orphan_labels))

    if orphan_labels:
        bordercontacts[tkey] = ipl.getlabel(orphan_labels, keys=key, return_only=True)


def count_contacts(ipl, bordercontacts, key, onecontactkey, multiplecontactkeys):

    labels, counts = np.unique(ipl['border_locmax'], return_counts=True)

    lblcounts = dict(zip(labels, counts))

    for k, v in lblcounts.iteritems():
        ipl.logging('Found {} border contacts for label {}.', v, k)

    bordercontacts[onecontactkey] = ipl.getlabel(list(labels[counts == 1]), keys=key, return_only=True)

    bordercontacts[multiplecontactkeys] = ipl.getlabel(list(labels[counts > 1]), keys=key, return_only=True)


def find_border_contacts(ipl, keys, thisparams):
    """
    :param ipl:

    ipl.get_params()

        find_border_contacts
            anisotropy
            return_bordercontact_images

        bordercontactsnames
        locmaxbordernames
            - 'border_locmax_1'
              ...
            - 'border_locmax_N'
            - 'disttransf_1'
              ...
            - 'disttransf_N'

    :param keys: list of source keys of length = N

    """

    N = len(keys)
    disttransfkeys = np.array(params['locmaxbordernames'])[N:]
    locmaxkeys = np.array(params['locmaxbordernames'])[:N]
    orphankeys = np.array(params['bordercontactsnames'][:N])
    onecontactkeys = np.array(params['bordercontactsnames'][N:2*N])
    multiplecontactkeys = np.array(params['bordercontactsnames'][2*N:])


    ipl.logging('locmaxkeys = {}', disttransfkeys)
    bordercontacts = IPL()

    c = 0
    for key in keys:

        ipl.logging('Finding border contacts for key = {}', key)

        # # Clear objects at the border
        # ipl.anytask(clear_border)

        # Do it manually
        # --------------
        # Compute the distance transform
        boundary_disttransf(ipl, thisparams, key, disttransfkeys[c])

        # For each of the 6 faces compute the objects which are touching it and the corresponding local maxima of the
        # distance transform
        compute_faces(ipl, (key, disttransfkeys[c]), 'faces')

        if thisparams['return_bordercontact_images']:
            # Use the faces to detect orphans
            find_orphans(ipl, bordercontacts, key, orphankeys[c])

        # Find global maxima for each object touching the border
        shp = ipl[key].shape
        ipl[locmaxkeys[c]] = np.zeros(shp)

        shps = {'xyf': shp[2],
                'xyb': shp[2],
                'xzf': shp[1],
                'xzb': shp[1],
                'yzf': shp[0],
                'yzb': shp[0]}
        # Define the relevant areas within the faces images for improved efficiency
        # Only labels found within these areas are checked for their maximum in the loops below
        areas = {'xyf': np.s_[shp[2]:shp[2]+shp[0], shp[2]:shp[2]+shp[1]],
                 'xyb': np.s_[shp[2]:shp[2]+shp[0], shp[2]:shp[2]+shp[1]],
                 'xzf': np.s_[shp[1]:shp[1]+shp[0], shp[1]:shp[1]+shp[2]],
                 'xzb': np.s_[shp[1]:shp[1]+shp[0], shp[1]:shp[1]+shp[2]],
                 'yzf': np.s_[shp[0]:shp[0]+shp[1], shp[0]:shp[0]+shp[2]],
                 'yzb': np.s_[shp[0]:shp[0]+shp[1], shp[0]:shp[0]+shp[2]]}

        # Do the computation for the merged and unmerged case
        find_border_centroids(ipl, shps, areas, key, disttransfkeys[c], locmaxkeys[c])

        count_contacts(ipl, bordercontacts, key, onecontactkeys[c], multiplecontactkeys[c])

        ipl['overlay{}'.format(c)] = np.array([(ipl[locmaxkeys[c]] > 0).astype(np.float32), (ipl[key].astype(np.float32)/np.amax(ipl[key])).astype(np.float32), (ipl[disttransfkeys[c]]/np.amax(ipl[disttransfkeys[c]])).astype(np.float32)])

        # ipl.pop(key)
        del ipl[key]

        c += 1

    del (ipl['faces'])

    if thisparams['return_bordercontact_images']:
        return [ipl, bordercontacts]
    else:
        return ipl


def find_border_contacts_image_iteration(ipl):
    params = ipl.get_params()
    thisparams = params['find_border_contacts']

    if thisparams['return_bordercontact_images']:
        bordercontacts = IPL()

    for d, k, v, kl in ipl.data_iterator(yield_short_kl=True):

        if k == params['largeobjname']:

            if thisparams['return_bordercontact_images']:
                ipl[kl].setlogger(ipl.getlogger())
                [ipl[kl], bordercontacts[kl]] = find_border_contacts(ipl[kl], (params['largeobjname'], params['largeobjmnames'][0]),
                                     thisparams)
            else:
                ipl[kl].setlogger(ipl.getlogger())
                ipl[kl] = find_border_contacts(ipl[kl], (params['largeobjname'], params['largeobjmnames'][0]),
                                     thisparams)

    if thisparams['return_bordercontact_images']:
        return bordercontacts
    else:
        return None


if __name__ == '__main__':

    resultsfolder = '/mnt/localdata02/jhennies/neuraldata/results/cremi_2016/161107_random_forest_of_paths/'

    yamlfile = resultsfolder + '/parameters.yml'

    ipl = IPL(
        yaml=yamlfile,
        yamlspec={'path': 'intermedfolder', 'filename': 'largeobjfile', 'skeys': 'largeobjname'},
        recursive_search=True        
    )
    params = ipl.get_params()
    thisparams = params['find_border_contacts']
    ipl.data_from_file(params['intermedfolder'] + params['largeobjmfile'],
                       skeys=params['largeobjmnames'][0], recursive_search=True, integrate=True)
    ipl.startlogger(filename=params['resultfolder'] + 'find_orphans.log', type='w')

    try:

        # Copy the script file and the parameters to the scriptsfolder
        copy(inspect.stack()[0][1], params['scriptsfolder'])
        copy(yamlfile, params['scriptsfolder'] + 'find_orphans.parameters.yml')
        # Write script and parameters to the logfile
        ipl.code2log(inspect.stack()[0][1])
        ipl.logging('')
        ipl.yaml2log()
        ipl.logging('')

        ipl.logging('\nipl datastructure: \n---\n{}', ipl.datastructure2string(maxdepth=3))

        bordercontacts = find_border_contacts_image_iteration(ipl)

        ipl.write(filepath=params['intermedfolder'] + params['locmaxborderfile'])
        if thisparams['return_bordercontact_images']:
            bordercontacts.write(filepath=params['intermedfolder'] + params['bordercontactsfile'])

        ipl.logging('\nFinal ipl dictionary structure:\n---\n{}', ipl.datastructure2string())
        if thisparams['return_bordercontact_images']:
            ipl.logging('\nFinal bordercontacts dictionary structure:\n---\n{}', bordercontacts.datastructure2string())

        ipl.logging('')
        ipl.stoplogger()

    except:

        ipl.errout('Unexpected error')