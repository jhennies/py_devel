from scipy import sparse, ndimage as ndi
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import vigra
from vigra import graphs

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

    def __init__(self):
        pass

    def set_data(self, data):
        self._data = data

    def calculate_superpixels(self, slicWeight=10, superpixelDiameter=10):

        img_lab = vigra.colors.transform_RGB2Lab(self._data)
        superpixels, nseg = vigra.analysis.slicSuperpixels(img_lab, slicWeight,
                                                      superpixelDiameter)
        superpixels = vigra.analysis.labelImage(superpixels)
        self._img_lab = img_lab
        self._superpixels = superpixels

    def calculate_rag(self, sigma_grad_mag=3.0):

        # compute gradient on interpolated image
        imgLabBig = vigra.resize(self._img_lab, [self._img_lab.shape[0]*2-1, self._img_lab.shape[1]*2-1])
        gradMag = vigra.filters.gaussianGradientMagnitude(imgLabBig, sigma_grad_mag)

        # get 2D grid graph and edgeMap for grid graph
        # from gradMag of interpolated image
        gridGraph = graphs.gridGraph(self._data.shape[0:2])
        gridGraphEdgeIndicator = graphs.edgeFeaturesFromInterpolatedImage(gridGraph,
                                                                          gradMag)
        self._grid_graph_edge_indicator = gridGraphEdgeIndicator

        # get region adjacency graph from super-pixel labels
        rag = graphs.regionAdjacencyGraph(gridGraph, labels)

        # accumulate edge weights from gradient magnitude
        ragEdgeIndicator = rag.accumulateEdgeFeatures(gridGraphEdgeIndicator)

        self._rag = rag
        self._rag_edge_indicator = ragEdgeIndicator

    def _buildAB(self, lap_sparse, labels):
        """
        Build the matrix A and rhs B of the linear system to solve.
        A and B are two block of the laplacian of the image graph.
        """

        # Adaptation: labels should already be passed in the correct format
        # labels = labels[labels >= 0]

        indices = np.arange(labels.size)
        unlabeled_indices = indices[labels == 0]
        seeds_indices = indices[labels > 0]
        # The following two lines take most of the time in this function
        B = lap_sparse[unlabeled_indices][:, seeds_indices]
        lap_sparse = lap_sparse[unlabeled_indices][:, unlabeled_indices]
        nlabels = labels.max()
        rhs = []
        for lab in range(1, nlabels + 1):
            mask = (labels[seeds_indices] == lab)
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

    def _build_laplacian(self, data, spacing, mask=None, beta=50, multichannel=False):

        pass
        # TODO: This has to be translated
        # l_x, l_y, l_z = tuple(data.shape[i] for i in range(3))
        # edges = _make_graph_edges_3d(l_x, l_y, l_z)
        # weights = _compute_weights_3d(data, spacing, beta=beta, eps=1.e-10,
        #                               multichannel=multichannel)
        # if mask is not None:
        #     edges, weights = _mask_edges_weights(edges, weights, mask)
        # lap = _make_laplacian_sparse(edges, weights)
        # del edges, weights
        # return lap

    def walker(self):
        pass
        # TODO: Load data as rag
        # TODO: Compute weights
        # TODO: Define seeds (labels)

        # TODO: _build_laplacian(...) has to be adapted
        # lap_sparse = self._build_laplacian(data, spacing, beta=self._beta, multichannel=self._multichannel)
        # TODO: make sure labels in in the correct format [l1, l2, ... ,ln] according to the rag
        # lap_sparse, B = self._buildAB(lap_sparse, labels)
        # X = self._solve_bf(lap_sparse, B, return_full_prob=self._return_full_prob)


if __name__ == "__main__":

    rsw = RandomSuperpixelWalker()

    # TODO: Load or create an image
    img = []

    rsw.set_data(img)
    rsw.calculate_superpixels()
    rsw.calculate_rag()

    rsw.walker()


    # data = np.zeros((5, 5))
    # data[3:5, 3:5] = 1
    # data += 0.35 * np.random.randn(*data.shape)
    #
    # data = np.atleast_3d(data)[..., np.newaxis]
    # print data
    #
    # spacing = np.asarray((1.,) * 3)
    # print spacing
    #
    # labels = np.zeros(data.shape, dtype=np.int)
    # labels[1, 1] = 1
    # labels[3, 3] = 2
    #
    # print labels
    #
    # lap_sparse = _build_laplacian(data, spacing, mask=labels >= 0,
    #                               beta=130, multichannel=False)
    #
    # print lap_sparse
    #
    # print "_buildAB()"
    # # labels = np.atleast_3d(labels)
    # lap_sparse, B = _buildAB(lap_sparse, labels)
    # print lap_sparse
    # print B
    #
    # print "test:"
    # #
    # # l_x, l_y, l_z = tuple(data.shape[i] for i in range(3))
    # # edges = _make_graph_edges_3d(l_x, l_y, l_z)
    # # print edges
    # #
    # # weights = _compute_weights_3d(data, spacing, beta=130, eps=1.e-10,
    # #                               multichannel=False)
    # #
    # # print weights
    # # print len(weights)
    # #
    # # # if mask is not None:
    # # #     edges, weights = _mask_edges_weights(edges, weights, mask)
    # # lap = _make_laplacian_sparse(edges, weights)
    # #
    # # print lap
    # #
    # # print "_make_laplacian_sparse(...)"
    # #
    # # pixel_nb = edges.max() + 1
    # # print pixel_nb
    # #
    # # diag = np.arange(pixel_nb)
    # # print diag
    # #
    # # i_indices = np.hstack((edges[0], edges[1]))
    # # j_indices = np.hstack((edges[1], edges[0]))
    # # print i_indices
    # # print j_indices
    # #
    # # lap_data = np.hstack((-weights, -weights))
    # #
    # # lap = sparse.coo_matrix((lap_data, (i_indices, j_indices)),
    # #                         shape=(pixel_nb, pixel_nb))
    # # print lap
    # #
    # # connect = - np.ravel(lap.sum(axis=1))
    # # print connect
    # #
    # # lap = sparse.coo_matrix(
    # #     (np.hstack((lap_data, connect)), (np.hstack((i_indices, diag)),
    # #                                   np.hstack((j_indices, diag)))),
    # #     shape=(pixel_nb, pixel_nb))
    # #
    # # print lap.tocsr()
    #
    #
    #
    # # del edges, weights
    # # return lap
    #
    # # _build_laplacian(data, spacing)
    #
    # # from skimage.segmentation import random_walker
    # # # from skimage.data import binary_blobs
    # # import skimage
    # #
    # # # Generate noisy synthetic data
    # # # data = skimage.img_as_float(binary_blobs(length=128, seed=1))
    # # data = np.zeros((100, 100))
    # # data[50:60, 50:70] = 1
    # # data += 0.35 * np.random.randn(*data.shape)
    # # markers = np.zeros(data.shape, dtype=np.uint)
    # # markers[data < -0.3] = 1
    # # markers[data > 1.3] = 2
    # #
    # # # Run random walker algorithm
    # # labels = random_walker(data, markers, beta=10, mode='bf')
    # #
    # # # Plot results
    # # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
    # #                                     sharex=True, sharey=True)
    # # ax1.imshow(data, cmap='gray', interpolation='nearest')
    # # ax1.axis('off')
    # # ax1.set_adjustable('box-forced')
    # # ax1.set_title('Noisy data')
    # # ax2.imshow(markers, cmap='hot', interpolation='nearest')
    # # ax2.axis('off')
    # # ax2.set_adjustable('box-forced')
    # # ax2.set_title('Markers')
    # # ax3.imshow(labels, cmap='gray', interpolation='nearest')
    # # ax3.axis('off')
    # # ax3.set_adjustable('box-forced')
    # # ax3.set_title('Segmentation')
    # #
    # # fig.tight_layout()
    # # plt.show()
