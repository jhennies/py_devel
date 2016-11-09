
from hdf5_image_processing import Hdf5ImageProcessingLib as IPL
from image_processing import ImageFileProcessing
from processing_lib import getlabel
from hdf5_processing import Hdf5Processing
import random
import vigra
import numpy as np
import os
import vigra.graphs as graphs
import sys
import traceback
import inspect
from shutil import copy
import processing_lib as lib


__author__ = 'jhennies'


def find_shortest_path(ipl, penaltypower, bounds, disttransf, locmax,
                       labels, labelgroup):

    # Modify distancetransform
    #
    # a) Invert: the lowest values (i.e. the lowest penalty for the shortest path detection) should be at the center of
    #    the current process
    disttransf = lib.invert_image(disttransf)
    #
    # b) Set all values outside the process to infinity
    disttransf = lib.filter_values(disttransf, np.amax(disttransf), type='eq', setto=np.inf)
    #
    # c) Increase the value difference between pixels near the boundaries and pixels central within the processes
    #    This increases the likelihood of the paths to follow the center of processes, thus avoiding short-cuts
    disttransf = lib.power(disttransf, penaltypower)

    # The situation:
    # We have multiple objects (n > 1) of unknown number.
    # We want to extract the local maxima within each object individually and create a
    #   list of all possible partners (pairs)
    # Each partner of a pair has to be located within a different object (label area)
    #
    # Approach 1:
    #   For each locmax iterate over all other locmaxs and write pairs, which satisfy the
    #   desired condition, to the pairs list
    #
    # Approach 2:
    #   For each label object iterate over all the others and append all possible locmax
    #   pairs to the pairs list
    #   Probably faster than approach 1 when implemented correctly? Someone should test that...

    # Approach 2 in its implemented form
    pairs = []
    for i in xrange(0, len(labelgroup)-1):
        indices_i = np.where((labels == labelgroup[i]) & (locmax > 0))
        indices_i = zip(indices_i[0], indices_i[1], indices_i[2])
        if indices_i:
            for j in xrange(i+1, len(labelgroup)):
                indices_j = np.where((labels == labelgroup[j]) & (locmax > 0))
                indices_j = zip(indices_j[0], indices_j[1], indices_j[2])
                if indices_j:
                    ipl.logging('Ind_i = {}\nInd_j = {}', indices_i, indices_j)
                    # Now, lets do some magic!
                    pairs = pairs + zip(indices_i * len(indices_j), sorted(indices_j * len(indices_i)))

    paths, pathim = lib.shortest_paths(disttransf, pairs, bounds=bounds, hfp=ipl)

    return paths, pathim


def paths_of_partners(ipl, lblsmkey, lblskey, changehashkey, locmaxkeys, disttransfkey, thisparams, ignore):

    if type(locmaxkeys) is str:
        locmaxkeys = (locmaxkeys,)

    # This results in the format:
    # more_keys = (locmaxkeys[0], ..., locmaxkeys[n], disttransfkey, lblskey)
    more_keys = locmaxkeys + (disttransfkey,) + (lblskey,)

    paths = IPL()
    for k in locmaxkeys:
        paths['pathsim', k] = np.zeros(ipl[lblsmkey].shape)

    # Determine labellist from change_hash (keys are strings due to h5 saving)
    labellist = ipl[changehashkey].keys()
    labellist = [int(x) for x in labellist]

    for lbl, lblim, more_ims, bounds in ipl.label_image_bounds_iterator(
        key=lblsmkey, background=0, more_keys=more_keys,
        maskvalue=0, value=0, labellist=labellist
    ):
        if lbl in ignore:
            continue
        # The format of more_ims is:
        # more_ims = {locmaxkeys[0]: locmax_1,
        #                ...
        #             locmaxkeys[n]: locmax_n,
        #             disttransfkey: disttransf,
        #             lblskey: lables}

        ipl.logging('======================\nWorking on label = {}', lbl)
        # ipl.logging('more_ims structure:\n---\n{}', more_ims.datastructure2string())
        ipl.logging('bounds = {}', bounds)

        for i in xrange(0, len(locmaxkeys)):

            ipl.logging('locmaxkey = {}', locmaxkeys[i])
            if np.amax(more_ims[locmaxkeys[i]]) > 0:
                ps, pathsim = find_shortest_path(ipl, thisparams['penaltypower'], bounds,
                                                 more_ims[disttransfkey],
                                                 more_ims[locmaxkeys[i]],
                                                 more_ims[lblskey],
                                                 ipl[changehashkey][str(lbl)])

                # Only store the path if the path-calculation successfully determined a path
                # Otherwise an empty list would be stored
                if ps:
                    ipl.logging('Number of paths found: {}', len(ps))

                    paths[lblsmkey, locmaxkeys[i], 'path', lbl] = ps
                    paths[lblsmkey, locmaxkeys[i], 'pathsim'] = pathsim

                    paths['pathsim', locmaxkeys[i]][bounds][pathsim > 0] = pathsim[pathsim > 0]

    for k in locmaxkeys:
        paths['overlay', k] = np.array([paths['pathsim', k],
                                     ipl[lblskey].astype(np.float32) / np.amax(ipl[lblskey]),
                                     vigra.filters.multiBinaryDilation(ipl[k].astype(np.uint8), 5)])

    return paths


def paths_of_partners_image_iteration(ipl):

    params = ipl.get_params()
    thisparams = params['paths_of_partners']

    paths = IPL()

    for d, k, v, kl in ipl.data_iterator(yield_short_kl=True):

        if k == params['locmaxbordernames'][1]:

            ipl[kl].setlogger(ipl.getlogger())
            paths[kl] = paths_of_partners(
                ipl[kl],
                params['largeobjmnames'][0], params['largeobjname'],
                params['largeobjmnames'][4],
                (params['locmaxbordernames'][1], params['locmaxnames'][1]),
                params['locmaxbordernames'][3],
                thisparams, thisparams['ignore']
            )

            # (ipl, 'largeobjm', 'largeobj',
            #  'change_hash', ('border_locmax_m', 'locmaxm'),
            #  'disttransfm', thisparams['ignore'])

    return paths


if __name__ == '__main__':

    resultsfolder = '/mnt/localdata02/jhennies/neuraldata/results/cremi_2016/161107_random_forest_of_paths/'

    yamlfile = resultsfolder + '/parameters.yml'

    ipl = IPL(
        yaml=yamlfile,
        yamlspec={'path': 'intermedfolder', 'filename': 'locmaxborderfile', 'skeys': {'locmaxbordernames': (1, 3)}},
        recursive_search=True
    )
    params = ipl.get_params()
    thisparams = params['paths_of_partners']
    ipl.startlogger(filename=params['resultfolder'] + 'paths_of_partners.log', type='w')
    ipl.data_from_file(params['intermedfolder'] + params['largeobjfile'],
                       skeys=params['largeobjname'],
                       recursive_search=True, integrate=True)
    ipl.data_from_file(params['intermedfolder'] + params['largeobjmfile'],
                       skeys=(params['largeobjmnames'][0], params['largeobjmnames'][4]),
                       recursive_search=True, integrate=True)
    ipl.data_from_file(params['intermedfolder'] + params['locmaxfile'],
                       skeys=params['locmaxnames'][1],
                       recursive_search=True, integrate=True)

    try:

        # Copy the script file and the parameters to the scriptsfolder
        copy(inspect.stack()[0][1], params['scriptsfolder'])
        copy(yamlfile, params['scriptsfolder'] + 'paths_of_partners.parameters.yml')
        # Write script and parameters to the logfile
        ipl.code2log(inspect.stack()[0][1])
        ipl.logging('')
        ipl.yaml2log()
        ipl.logging('')

        ipl.logging('\nipl datastructure: \n\n{}', ipl.datastructure2string(maxdepth=3))

        paths = paths_of_partners_image_iteration(ipl)

        paths.write(filepath=params['intermedfolder'] + params['pathsfalsefile'])

        ipl.logging('\nFinal dictionary structure:\n---\n{}', paths.datastructure2string())
        ipl.logging('')
        ipl.stoplogger()

    except:

        ipl.errout('Unexpected error')
