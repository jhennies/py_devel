from scipy import sparse
from scipy.sparse import linalg
import numpy as np
import vigra
from vigra import graphs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mplib
from random import random

class RandomSuperpixelWalker:

    _return_full_prob = False
    _beta = 130
    _multichannel = False
    _data = []
    _superpixels = []
    _labels = []
    _img_lab = []
    _grid_graph_edge_indicator = None
    _rag_edge_indicator = None
    _rag = None
    _edge_uv_ids = None
    _edge_weights = None
    _seeds = None
    _B = None
    _lap_sparse = None

    def __init__(self):
        pass

    def set_beta(self, beta):
        self._beta = beta

    def set_data(self, data):
        self._data = data

    def set_superpixels(self, superpixels):
        self._superpixels = superpixels

    def calc_slic_superpixels(self, slicWeight=10, superpixelDiameter=10):

        # img_lab = vigra.colors.transform_RGB2Lab(self._data)
        # img_lab = vigra.analysis.labelImage(self._data)

        # superpixels, nseg = vigra.analysis.slicSuperpixels(img_lab, slicWeight,
        #                                               superpixelDiameter)

        # print "self._data.shape"
        # print self._data.shape
        # print "self._data.dtype"
        # print self._data.dtype
        superpixels, nseg = vigra.analysis.slicSuperpixels(self._data, slicWeight,
                                                      superpixelDiameter)
        # print "superpixels"
        # print superpixels
        # print superpixels.shape
        # superpixels = vigra.analysis.labelImage(superpixels)
        # self._img_lab = img_lab
        self._superpixels = superpixels

    def calc_rag(self, sigma_grad_mag=3.0):

        # # compute gradient on interpolated image
        # imgLabBig = vigra.resize(self._img_lab, [self._img_lab.shape[0]*2-1, self._img_lab.shape[1]*2-1])
        # gradMag = vigra.filters.gaussianGradientMagnitude(imgLabBig, sigma_grad_mag)
        #
        # # get 2D grid graph and edgeMap for grid graph
        # # from gradMag of interpolated image
        # gridGraph = graphs.gridGraph(self._data.shape[0:2])
        # gridGraphEdgeIndicator = graphs.edgeFeaturesFromInterpolatedImage(gridGraph,
        #                                                                   gradMag)
        # self._grid_graph_edge_indicator = gridGraphEdgeIndicator
        #
        # # get region adjacency graph from super-pixel labels
        # rag = graphs.regionAdjacencyGraph(gridGraph, self._superpixels)

        (grag, rag) = graphs.gridRegionAdjacencyGraph(self._superpixels)

        # # accumulate edge weights from gradient magnitude
        # ragEdgeIndicator = rag.accumulateEdgeFeatures(gridGraphEdgeIndicator)

        self._rag = rag
        # self._rag_edge_indicator = ragEdgeIndicator

    def _buildAB(self, lap_sparse):
        """
        Build the matrix A and rhs B of the linear system to solve.
        A and B are two block of the laplacian of the image graph.
        """
        #
        # print "_buildAB"
        #
        # print "_seeds"
        # print self._seeds
        indices = np.arange(self._seeds.size)
        unlabeled_indices = indices[self._seeds == 0]
        seed_indices = indices[self._seeds > 0]

        # print "unlabeled_indices"
        # print unlabeled_indices
        # print "seed_indices"
        # print seed_indices
        # print "lap_sparse.shape"
        # print lap_sparse.shape
        B = lap_sparse[unlabeled_indices][:, seed_indices]
        lap_sparse = lap_sparse[unlabeled_indices][:, unlabeled_indices]

        n_seeds = self._seeds.max()
        # print "n_seeds"
        # print n_seeds
        rhs = []
        for seed in range(1, n_seeds + 1):
            mask = (self._seeds[seed_indices] == seed)
            fs = sparse.csr_matrix(mask)
            fs = fs.transpose()
            rhs.append(B * fs)

        return lap_sparse, rhs

    def _solve_bf(self, lap_sparse, B, return_full_prob=False):
        """
        solves lap_sparse X_i = B_i for each phase i. An LU decomposition
        of lap_sparse is computed first. For each pixel, the label i
        corresponding to the maximal X_i is returned.
        """
        lap_sparse = lap_sparse.tocsc()
        solver = sparse.linalg.factorized(lap_sparse.astype(np.double))
        X = np.array([solver(np.array((-B[i]).todense()).ravel())
                      for i in range(len(B))])
        if not self._return_full_prob:
            X = np.argmax(X, axis=0)
        return X

    def _compute_weights(self, eps=1.e-10):

        beta = self._beta / (10 * self._data.std())

        # Iterate over all edges...
        edge_num = self._rag.edgeNum
        # print "edge_num"
        # print edge_num
        weights = np.zeros((edge_num,))
        i = 0
        for edge in self._rag.edgeIter():

            eid = self._rag.id(edge)
            uc, vc = self._rag.edgeUVCoordinates(eid)

            # Get image intensities
            if len(self._data.shape) == 3:
                u_int = self._data[uc.transpose()[0], uc.transpose()[1], uc.transpose()[2]]
                v_int = self._data[vc.transpose()[0], vc.transpose()[1], vc.transpose()[2]]
            elif len(self._data.shape) == 2:
                u_int = self._data[uc.transpose()[0], uc.transpose()[1]]
                v_int = self._data[vc.transpose()[0], vc.transpose()[1]]

            # Calculate weights
            # --- In the original skimage version this was computed using the sum of the gradients rather than the mean
            # --- For superpixels I guess it makes more sense this way...
            weight = np.exp(- beta * np.mean(np.abs(u_int - v_int)))
            weights[i] = weight
            i += 1

        weights += eps

        # My first version...
        # # print "_compute_weights"
        #
        # uc = None
        # # Iterate over all edges
        #
        # edge_num = self._rag.edgeNum
        # # print "edge_num"
        # # print edge_num
        # weights = np.zeros((edge_num,))
        # i = 0
        # for edge in self._rag.edgeIter():
        #
        #     eid = self._rag.id(edge)
        #     uc, vc = self._rag.edgeUVCoordinates(eid)
        #
        #     # Get image intensities
        #     if len(self._data.shape) == 3:
        #         u_int = self._data[uc.transpose()[0], uc.transpose()[1], uc.transpose()[2]]
        #         v_int = self._data[vc.transpose()[0], vc.transpose()[1], vc.transpose()[2]]
        #     elif len(self._data.shape) == 2:
        #         u_int = self._data[uc.transpose()[0], uc.transpose()[1]]
        #         v_int = self._data[vc.transpose()[0], vc.transpose()[1]]
        #
        #     # Calculate weights
        #     # weight = self.beta * (np.mean(u_int) + np.mean(v_int) / 2)
        #     weight = np.power(np.e, -self._beta * np.power(np.mean(u_int) - np.mean(v_int), 2))
        #     # weight = np.power(np.e, -self._beta * np.power(np.mean(u_int) + np.mean(v_int), 2))
        #     # print weight
        #     # print "weight"
        #     # print weight
        #     weights[i] = weight
        #     i += 1
        #
        # # print "uc"
        # # print uc
        # # print self._data[uc.transpose()[0], uc.transpose()[1], uc.transpose()[2]]
        #
        # # print "weights"
        # # print weights
        #
        # # for i in rsw.get_rag().edgeIter():
        return weights

    def _build_laplacian(self):

        # print "_build_laplacian"

        # Edges
        self._edge_uv_ids = rsw.get_rag().uvIds().transpose()
        # print "_edge_uv_ids"
        # print self._edge_uv_ids
        # print self._edge_uv_ids.max()

        # Weights
        self._edge_weights = self._compute_weights()

        lap = self._make_laplacian_sparse()
        # print "lap"
        # print lap

        return lap

    def _make_laplacian_sparse(self):

        # print "_make_laplacian_sparse"

        pixel_nb = self._edge_uv_ids.max() + 1
        # print "pixel_nb"
        # print pixel_nb

        diag = np.arange(pixel_nb)
        i_indices = np.hstack((self._edge_uv_ids[0], self._edge_uv_ids[1]))
        j_indices = np.hstack((self._edge_uv_ids[1], self._edge_uv_ids[0]))
        # print "i_indices"
        # print i_indices
        # print i_indices.shape
        # print "j_indices"
        # print j_indices
        # print j_indices.shape

        weights = np.hstack((-self._edge_weights, -self._edge_weights))
        # print "weights"
        # print weights
        # print weights.shape

        lap = sparse.coo_matrix((weights, (i_indices, j_indices)),
                                shape=(pixel_nb, pixel_nb))
        connect = - np.ravel(lap.sum(axis=1))
        lap = sparse.coo_matrix(
            (np.hstack((weights, connect)), (np.hstack((i_indices, diag)),
                                          np.hstack((j_indices, diag)))),
            shape=(pixel_nb, pixel_nb))

        # print "lap.tocsr"
        # print lap.tocsr()

        return lap.tocsr()

    def get_rag(self):
        return self._rag

    def set_seeds(self, seeds):
        self._seeds = seeds

    def init_seeds(self, zero_based=False):
        # print "rag.nodeNum"
        # print self._rag.nodeNum
        if zero_based:
            self._seeds = np.zeros((self._rag.nodeNum,), dtype=np.uint32)
        else:
            self._seeds = np.zeros((self._rag.nodeNum + 1,), dtype=np.uint32)

    def set_seed(self, seed, value):
        self._seeds[seed] = value

    def walker(self, mode='bf', return_full_prob=False, tol=1.e-3):

        # self._data = np.atleast_3d(data)[..., np.newaxis]

        lap_sparse = self._build_laplacian()
        # print "lap_sparse"
        # print lap_sparse

        lap_sparse, B = self._buildAB(lap_sparse)
        # print "lap_sparse"
        # print lap_sparse
        # print "B"
        # print B
        self._B = B
        self._lap_sparse = lap_sparse

        if mode == 'cg':
            X = self._solve_cg(lap_sparse, B, tol=tol, return_full_prob=return_full_prob)
        # TODO
        # if mode == 'cg_mg':
        #     if not amg_loaded:
        #         warnings.warn(
        #             """pyamg (http://pyamg.org/)) is needed to use
        #             this mode, but is not installed. The 'cg' mode will be used
        #             instead.""")
        #         X = _solve_cg(lap_sparse, B, tol=tol,
        #                       return_full_prob=return_full_prob)
        #     else:
        #         X = _solve_cg_mg(lap_sparse, B, tol=tol,
        #                          return_full_prob=return_full_prob)
        if mode == 'bf':
            X = self._solve_bf(lap_sparse, B, return_full_prob=return_full_prob)

        # print "X"
        # print X
        # print X.shape

        result = self._seeds
        result[result == 0] = X + 1

        # print "result"
        # print result
        # print result.shape

        return result

    def _solve_bf(self, lap_sparse, B, return_full_prob=False):
        """
        solves lap_sparse X_i = B_i for each phase i. An LU decomposition
        of lap_sparse is computed first. For each pixel, the label i
        corresponding to the maximal X_i is returned.
        """
        lap_sparse = lap_sparse.tocsc()
        solver = sparse.linalg.factorized(lap_sparse.astype(np.double))
        X = np.array([solver(np.array((-B[i]).todense()).ravel())
                      for i in range(len(B))])
        if not return_full_prob:
            X = np.argmax(X, axis=0)
        return X

    def _solve_cg(self, lap_sparse, B, tol=1.e-3, return_full_prob=False):
        """
        solves lap_sparse X_i = B_i for each phase i, using the conjugate
        gradient method. For each pixel, the label i corresponding to the
        maximal X_i is returned.
        """
        lap_sparse = lap_sparse.tocsc()
        X = []
        for i in range(len(B)):
            x0 = linalg.isolve.iterative.cg(lap_sparse, -B[i].todense(), tol=tol)[0]
            X.append(x0)
        if not return_full_prob:
            X = np.array(X)
            X = np.argmax(X, axis=0)
        return X

if __name__ == "__main__":

    # EXAMPLE 1 ########################################################################################################

    img = vigra.impex.readHDF5("/windows/mobi/h1.hci/isbi_2013/data/test-input.crop_100_100_100.h5", "data")[:, :, 50]
    print img.shape
    rsw = RandomSuperpixelWalker()
    rsw.set_data(img)
    rsw.set_beta(130)

    rsw.calc_slic_superpixels(slicWeight=0.15, superpixelDiameter=10)

    rsw.calc_rag()

    rsw.init_seeds()
    rsw.set_seed(17, 1)
    rsw.set_seed(85, 2)

    result = rsw.walker(return_full_prob=False, mode='cg')

    f = plt.figure(figsize=(8, 4))

    ax1 = f.add_subplot(131)
    ax1.imshow(img, cmap=cm.Greys_r)
    ax1.set_title("Image")
    plt.axis('off')

    ax2 = f.add_subplot(133)
    n_result = np.zeros((result.shape[0]+1,), dtype=result.dtype)
    n_result[0:-1] = result
    resImg = rsw.get_rag().projectLabelsToGridGraph(n_result)
    ax2.imshow(resImg)
    ax2.set_title("Result-Segmentation")
    plt.axis('off')

    ax3 = f.add_subplot(132)
    print result.shape
    segm = np.zeros(result.shape, dtype=result.dtype)
    segm[:] = range(0, result.shape[0])
    rsw.get_rag().projectLabelsToGridGraph(segm)
    colors = [(1, 1, 1)] + [(random(), random(), random()) for i in xrange(255)]
    new_map = mplib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)
    ax3.imshow(rsw.get_rag().projectLabelsToGridGraph(segm), cmap=new_map)
    ax3.set_title("Superpixels")
    plt.axis('off')

    mplib.pyplot.draw()
    vigra.show()


    # # EXAMPLE 2 ########################################################################################################
    # # Comparison to skimage random walker
    #
    # # Generate noisy synthetic data
    # # -----------------------------
    #
    # data = np.zeros((100, 100), dtype=np.float64)
    # data[50:60, 50:70] = 1
    # data[20:40, 20:30] = 1
    # data[30:40, 80:90] = 1
    # data += 0.35 * np.random.randn(*data.shape)
    # print data.dtype
    # markers = np.zeros(data.shape, dtype=np.uint32)
    # markers[data < -0.3] = 1
    # markers[data > 1.3] = 2
    #
    # # Random superpixel walker...
    # # ---------------------------
    #
    # rsw = RandomSuperpixelWalker()
    # rsw.set_data(data)
    # rsw.set_beta(130)
    #
    # # Create superpixels the size of one pixel (for comparison to the pixel-wise random walker)
    # sp = np.zeros((data.shape[0] * data.shape[1],), dtype=np.uint32)
    # sp[:] = range(0, sp.shape[0])
    # sp = np.reshape(sp, data.shape)
    # rsw.set_superpixels(sp)
    #
    # rsw.calc_rag()
    #
    # seeds = np.reshape(markers.copy(), (markers.shape[0] * markers.shape[1],))
    # rsw.set_seeds(seeds)
    # rsw_result = rsw.walker(return_full_prob=False)
    # rsw_res_img = rsw.get_rag().projectLabelsToGridGraph(rsw_result)
    # print rsw_result.shape
    #
    # # Skimage random walker...
    # # ------------------------
    #
    # from skimage.segmentation import random_walker
    #
    # # Run random walker algorithm
    # rw_result = random_walker(data, markers, beta=10, mode='bf')
    #
    # # Plot results
    # # ------------------------
    #
    # fig, axarr = plt.subplots(2, 2, figsize=(8, 8),
    #                                     sharex=True, sharey=True)
    # axarr[0, 0].imshow(data, cmap='gray', interpolation='nearest')
    # axarr[0, 0].axis('off')
    # axarr[0, 0].set_adjustable('box-forced')
    # axarr[0, 0].set_title('Noisy data')
    # axarr[0, 1].imshow(markers, cmap='hot', interpolation='nearest')
    # axarr[0, 1].axis('off')
    # axarr[0, 1].set_adjustable('box-forced')
    # axarr[0, 1].set_title('Markers')
    # axarr[1, 0].imshow(rw_result, cmap='gray', interpolation='nearest')
    # axarr[1, 0].axis('off')
    # axarr[1, 0].set_adjustable('box-forced')
    # axarr[1, 0].set_title('Skimage RW Segmentation')
    # axarr[1, 1].imshow(rsw_res_img, cmap='gray', interpolation='nearest')
    # axarr[1, 1].axis('off')
    # axarr[1, 1].set_adjustable('box-forced')
    # axarr[1, 1].set_title('RSW segmentation')
    #
    # fig.tight_layout()
    # plt.show()
