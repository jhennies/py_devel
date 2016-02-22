from scipy import sparse, ndimage as ndi
from scipy.sparse import linalg
from image_processing.make_membrane_label import ImageFileProcessing
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import vigra
from vigra import graphs
from skimage.segmentation import random_walker
import matplotlib.pyplot as plt
import scipy
import skimage
import matplotlib.cm as cm

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

        print "self._data.shape"
        print self._data.shape
        print "self._data.dtype"
        print self._data.dtype
        superpixels, nseg = vigra.analysis.slicSuperpixels(self._data, slicWeight,
                                                      superpixelDiameter)
        print "superpixels"
        # print superpixels
        print superpixels.shape
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

        print "_buildAB"

        print "_seeds"
        print self._seeds
        indices = np.arange(self._seeds.size)
        unlabeled_indices = indices[self._seeds == 0]
        seed_indices = indices[self._seeds > 0]

        print "unlabeled_indices"
        print unlabeled_indices
        print "seed_indices"
        print seed_indices
        print "lap_sparse.shape"
        print lap_sparse.shape
        B = lap_sparse[unlabeled_indices][:, seed_indices]
        lap_sparse = lap_sparse[unlabeled_indices][:, unlabeled_indices]

        n_seeds = self._seeds.max()
        print "n_seeds"
        print n_seeds
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

    def _compute_weights(self):

        print "_compute_weights"

        uc = None
        # Iterate over all edges

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
            # weight = self.beta * (np.mean(u_int) + np.mean(v_int) / 2)
            weight = np.power(np.e, -self._beta * np.power(np.mean(u_int) - np.mean(v_int), 2))
            # weight = np.power(np.e, -self._beta * np.power(np.mean(u_int) + np.mean(v_int), 2))
            print weight
            # print "weight"
            # print weight
            weights[i] = weight
            i += 1

        # print "uc"
        # print uc
        # print self._data[uc.transpose()[0], uc.transpose()[1], uc.transpose()[2]]

        # print "weights"
        # print weights

        # for i in rsw.get_rag().edgeIter():
        return weights

    def _build_laplacian(self):

        print "_build_laplacian"

        # Edges
        self._edge_uv_ids = rsw.get_rag().uvIds().transpose()
        print "_edge_uv_ids"
        print self._edge_uv_ids
        print self._edge_uv_ids.max()

        # Weights
        self._edge_weights = self._compute_weights()

        lap = self._make_laplacian_sparse()
        print "lap"
        print lap

        return lap

    def _make_laplacian_sparse(self):

        print "_make_laplacian_sparse"

        pixel_nb = self._edge_uv_ids.max()+1
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

    def init_seeds(self):
        print "rag.nodeNum"
        print self._rag.nodeNum
        self._seeds = np.zeros((self._rag.nodeNum+1,), dtype=np.uint32)

    def set_seed(self, seed, value):
        self._seeds[seed] = value

    def walker(self, mode='bf', return_full_prob=False):

        lap_sparse = self._build_laplacian()
        print "lap_sparse"
        print lap_sparse

        lap_sparse, B = self._buildAB(lap_sparse)
        print "lap_sparse"
        print lap_sparse
        print "B"
        print B
        self._B = B
        self._lap_sparse = lap_sparse

        # TODO
        # if mode == 'cg':
        #     X = self._solve_cg(lap_sparse, B, tol=tol,
        #               return_full_prob=return_full_prob)
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

        print "X"
        print X
        print X.shape

        result = self._seeds
        result[result == 0] = X

        print "result"
        print result
        print result.shape

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

    # def _solve_cg(self, lap_sparse, B, tol, return_full_prob=False):
    #     """
    #     solves lap_sparse X_i = B_i for each phase i, using the conjugate
    #     gradient method. For each pixel, the label i corresponding to the
    #     maximal X_i is returned.
    #     """
    #     lap_sparse = lap_sparse.tocsc()
    #     X = []
    #     for i in range(len(B)):
    #         x0 = cg(lap_sparse, -B[i].todense(), tol=tol)[0]
    #         X.append(x0)
    #     if not return_full_prob:
    #         X = np.array(X)
    #         X = np.argmax(X, axis=0)
    #     return X

def _build_laplacian(data, spacing, mask=None, beta=50,
                     multichannel=False):
    l_x, l_y, l_z = tuple(data.shape[i] for i in range(3))
    edges = _make_graph_edges_3d(l_x, l_y, l_z)
    weights = _compute_weights_3d(data, spacing, beta=beta, eps=1.e-10,
                                  multichannel=multichannel)
    print edges
    print weights
    if mask is not None:
        edges, weights = _mask_edges_weights(edges, weights, mask)
    lap = _make_laplacian_sparse(edges, weights)
    # print lap
    del edges, weights
    return lap

def _make_laplacian_sparse(edges, weights):
    """
    Sparse implementation
    """
    print edges
    print weights
    pixel_nb = edges.max() + 1
    diag = np.arange(pixel_nb)
    i_indices = np.hstack((edges[0], edges[1]))
    j_indices = np.hstack((edges[1], edges[0]))
    data = np.hstack((-weights, -weights))
    lap = sparse.coo_matrix((data, (i_indices, j_indices)),
                            shape=(pixel_nb, pixel_nb))
    connect = - np.ravel(lap.sum(axis=1))
    lap = sparse.coo_matrix(
        (np.hstack((data, connect)), (np.hstack((i_indices, diag)),
                                      np.hstack((j_indices, diag)))),
        shape=(pixel_nb, pixel_nb))
    return lap.tocsr()

def _mask_edges_weights(edges, weights, mask):
    """
    Remove edges of the graph connected to masked nodes, as well as
    corresponding weights of the edges.
    """
    mask0 = np.hstack((mask[:, :, :-1].ravel(), mask[:, :-1].ravel(),
                       mask[:-1].ravel()))
    mask1 = np.hstack((mask[:, :, 1:].ravel(), mask[:, 1:].ravel(),
                       mask[1:].ravel()))
    ind_mask = np.logical_and(mask0, mask1)
    edges, weights = edges[:, ind_mask], weights[ind_mask]
    max_node_index = edges.max()
    # Reassign edges labels to 0, 1, ... edges_number - 1
    order = np.searchsorted(np.unique(edges.ravel()),
                            np.arange(max_node_index + 1))
    edges = order[edges.astype(np.int64)]
    return edges, weights

def _compute_weights_3d(data, spacing, beta=130, eps=1.e-6,
                        multichannel=False):
    # Weight calculation is main difference in multispectral version
    # Original gradient**2 replaced with sum of gradients ** 2
    gradients = 0
    for channel in range(0, data.shape[-1]):
        gradients += _compute_gradients_3d(data[..., channel],
                                           spacing) ** 2
    # All channels considered together in this standard deviation
    beta /= 10 * data.std()
    if multichannel:
        # New final term in beta to give == results in trivial case where
        # multiple identical spectra are passed.
        beta /= np.sqrt(data.shape[-1])
    gradients *= beta
    weights = np.exp(- gradients)
    weights += eps
    return weights

def _compute_gradients_3d(data, spacing):
    gr_deep = np.abs(data[:, :, :-1] - data[:, :, 1:]).ravel() / spacing[2]
    gr_right = np.abs(data[:, :-1] - data[:, 1:]).ravel() / spacing[1]
    gr_down = np.abs(data[:-1] - data[1:]).ravel() / spacing[0]
    return np.r_[gr_deep, gr_right, gr_down]

def _buildAB(lap_sparse, labels):
    """
    Build the matrix A and rhs B of the linear system to solve.
    A and B are two block of the laplacian of the image graph.
    """
    labels = labels[labels >= 0]
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

def _make_graph_edges_3d(n_x, n_y, n_z):
    """Returns a list of edges for a 3D image.

    Parameters
    ----------
    n_x: integer
        The size of the grid in the x direction.
    n_y: integer
        The size of the grid in the y direction
    n_z: integer
        The size of the grid in the z direction

    Returns
    -------
    edges : (2, N) ndarray
        with the total number of edges::

            N = n_x * n_y * (nz - 1) +
                n_x * (n_y - 1) * nz +
                (n_x - 1) * n_y * nz

        Graph edges with each column describing a node-id pair.
    """
    vertices = np.arange(n_x * n_y * n_z).reshape((n_x, n_y, n_z))
    edges_deep = np.vstack((vertices[:, :, :-1].ravel(),
                            vertices[:, :, 1:].ravel()))
    edges_right = np.vstack((vertices[:, :-1].ravel(),
                             vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))
    edges = np.hstack((edges_deep, edges_right, edges_down))
    return edges

def _solve_bf(lap_sparse, B, return_full_prob=False):
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

def _clean_labels_ar(X, labels, copy=False):
    X = X.astype(labels.dtype)
    if copy:
        labels = np.copy(labels)
    labels = np.ravel(labels)
    labels[labels == 0] = X
    return labels


if __name__ == "__main__":

    rsw = RandomSuperpixelWalker()
    # ifp = ImageFileProcessing()
    # img = ifp.load_h5("/windows/mobi/h1.hci/isbi_2013/data/train-input-crop50.h5", im_id=0)[:, :, 1]

    # print img

    # # Generate noisy synthetic data
    # # data = skimage.img_as_float(binary_blobs(length=128, seed=1))
    # img = np.zeros((100, 100), dtype=np.float32)
    # img[50:60, 50:70] = 1
    # img += 0.35 * np.random.randn(*img.shape)
    # markers = np.zeros(img.shape, dtype=np.uint)
    # markers[img < -0.3] = 1
    # markers[img > 1.3] = 2

    img = vigra.impex.readHDF5("/windows/mobi/h1.hci/isbi_2013/data/test-input.crop_100_100_100.h5", "data")[:, :, 50]
    # img = vigra.impex.readImage("/home/jhennies/src/vigra/vigranumpy/examples/100075.jpg")
    print img.shape
    rsw.set_data(img)
    rsw.set_beta(10000)

    rsw.calc_slic_superpixels(slicWeight=0.15, superpixelDiameter=10)
    # sp = np.zeros((img.shape[0]*img.shape[1],), dtype=np.uint32)
    # sp[:] = range(0, 10000)
    # print sp.shape
    # print img.shape
    # sp = sp.reshape(img.shape)
    # print sp.shape
    # rsw.set_superpixels(superpixels=sp)

    rsw.calc_rag()
    # print rsw.get_rag()

    rsw.init_seeds()
    rsw.set_seed(17, 1)
    rsw.set_seed(85, 2)
    # rsw.set_seed(8900, 2)

    result = rsw.walker(return_full_prob=False)

    f = plt.figure()

    ax1 = f.add_subplot(221)
    print rsw.get_rag()
    print result.dtype
    print result.shape
    # newResult = np.zeros((result.shape[0]+1,), dtype=result.dtype)
    # print newResult.shape
    # newResult[1:] = result
    # result = newResult
    # print result.shape
    print img.shape
    # result = np.zeros((25,), dtype=np.uint32)
    # result[:] = range(0, 25)
    # result[10:25] = 2
    n_result = np.zeros((result.shape[0]+1,), dtype=result.dtype)
    n_result[0:-1] = result
    resImg = rsw.get_rag().projectLabelsToGridGraph(n_result)
    ax1.imshow(resImg)
    ax1.set_title("Result-Segmentation")
    plt.axis('off')

    ax2 = f.add_subplot(222)
    ax2.imshow(img, cmap=cm.Greys_r)
    ax2.set_title("Image")
    plt.axis('off')

    ax3 = f.add_subplot(223)
    print result.shape
    segm = np.zeros(result.shape, dtype=result.dtype)
    segm[:] = range(0, result.shape[0])
    rsw.get_rag().projectLabelsToGridGraph(segm)
    ax3.imshow(rsw.get_rag().projectLabelsToGridGraph(segm))

    vigra.show()

    # print scipy.version.version

    # # Generate noisy synthetic data
    # # data = skimage.img_as_float(binary_blobs(length=128, seed=1))
    # data = np.zeros((100, 100), dtype=np.float32)
    # data[50:60, 50:70] = 1
    # data += 0.35 * np.random.randn(*data.shape)
    # markers = np.zeros(data.shape, dtype=np.uint)
    # markers[data < -0.3] = 1
    # markers[data > 1.3] = 2

    # # RandomSuperpixelWalker
    # rsw = RandomSuperpixelWalker()
    # rsw.set_data(data)
    # rsw.set_beta(0.1)
    # rsw.calculate_superpixels(10, 20)
    # rsw.calculate_rag()
    # rsw.init_seeds()
    # rsw.set_seed(1, 1)
    # rsw.set_seed(10, 2)
    # result = rsw.walker(return_full_prob=False)
    #
    # f = pylab.figure()
    # # ax1 = f.add_subplot(2, 2, 1)
    # # vigra.imshow(gradMag,show=False)
    # # ax1.set_title("Input Image")
    # # pylab.axis('off')
    # #
    # # ax2 = f.add_subplot(2, 2, 2)
    # # rag.show(img)
    # # ax2.set_title("Over-Segmentation")
    # # pylab.axis('off')
    # #
    # # ax3 = f.add_subplot(2, 2, 3)
    # # rag.show(img, labels)
    # # ax3.set_title("Result-Segmentation")
    # # pylab.axis('off')
    #
    # # ax4 = f.add_subplot(2, 2, 4)
    # print rsw._rag
    # print result.dtype
    # print result.shape
    # # newResult = np.zeros((result.shape[0]+1,), dtype=result.dtype)
    # # print newResult.shape
    # # newResult[1:] = result
    # # result = newResult
    # # print result.shape
    # print data.shape
    # result = np.zeros((25,), dtype=np.uint32)
    # result[:] = range(0, 25)
    # # result[10:25] = 2
    # rsw.get_rag().showNested(data, result)
    # f.set_title("Result-Segmentation")
    # pylab.axis('off')
    #
    # vigra.show()

    # # Run random walker algorithm
    # labels = random_walker(data, markers, beta=10, mode='bf')
    #
    # # Plot results
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
    #                                     sharex=True, sharey=True)
    # ax1.imshow(data, cmap='gray', interpolation='nearest')
    # ax1.axis('off')
    # ax1.set_adjustable('box-forced')
    # ax1.set_title('Noisy data')
    # ax2.imshow(markers, cmap='hot', interpolation='nearest')
    # ax2.axis('off')
    # ax2.set_adjustable('box-forced')
    # ax2.set_title('Markers')
    # ax3.imshow(labels, cmap='gray', interpolation='nearest')
    # ax3.axis('off')
    # ax3.set_adjustable('box-forced')
    # ax3.set_title('Segmentation')
    #
    # fig.tight_layout()
    # plt.show()







    # for i in rsw.get_rag().edgeIter():
    #     eid = rsw.get_rag().id(i)
    #     print "======"
    #     print "eid"
    #     print eid
    #     uc, vc = rsw.get_rag().edgeUVCoordinates(eid)
    #     # print vc
    #     ui = rsw.get_rag().u(i)
    #     # print ui
    #     print "uId"
    #     print rsw.get_rag().uId(i)
    #     print "vId"
    #     print rsw.get_rag().vId(i)
    #     print "uvId"
    #     print rsw.get_rag().uvId(i)

    # print "====="
    # print "uvIds"
    # print rsw.get_rag().uvIds()
    # print rsw.get_rag().uvIds().transpose()
    # print rsw.get_rag().uvIds().shape
    #
    # print "nodeIdMap"
    # print rsw.get_rag().nodeIdMap()



    #
    # data = np.zeros((5, 5))
    # data[3:5, 3:5] = 1
    # data += 0.35 * np.random.randn(*data.shape)
    #
    # data = np.atleast_3d(data)[..., np.newaxis]
    # # print data
    #
    # spacing = np.asarray((1.,) * 3)
    # # print spacing
    #
    # labels = np.zeros(data.shape, dtype=np.int)
    # labels[1, 1] = 1
    # labels[3, 3] = 2
    # labels[3, 4] = 2
    #
    #
    # # print labels
    #
    # lap_sparse = _build_laplacian(data, spacing, mask=labels >= 0,
    #                               beta=130, multichannel=False)
    #
    # # # print lap_sparse
    # # labels = labels[labels >= 0]
    # # print "labels"
    # # print labels
    # # indices = np.arange(labels.size)
    # # seed_indices = indices[labels > 0]
    # # print "seed_indices"
    # # print seed_indices
    # # unlabeled_indices = indices[labels == 0]
    # # print "unlabeled_indices"
    # # print unlabeled_indices
    # # print 'lap_sparse[unlabeled_indices]'
    # # print lap_sparse[unlabeled_indices]
    # # print 'lap_sparse[unlabeled_indices][:, seed_indices]'
    # # print lap_sparse[unlabeled_indices][:, seed_indices]
    #
    #
    # # print "_buildAB()"
    # # labels = np.atleast_3d(labels)
    # lap_sparse, B = _buildAB(lap_sparse, labels)
    # # print 'lap_sparse: '
    # # print lap_sparse
    # print 'B'
    # print B[0]
    # print B[1]
    #
    # return_full_prob = False
    # X = _solve_bf(lap_sparse, B,
    #                   return_full_prob=return_full_prob)
    #
    # print "X"
    # print X
    # print X.shape
    #
    # dims = data.shape
    #
    # if return_full_prob:
    #     labels = labels.astype(np.float)
    #     X = np.array([_clean_labels_ar(Xline, labels, copy=True).reshape(dims)
    #                   for Xline in X])
    #     for i in range(1, int(labels.max()) + 1):
    #         mask_i = np.squeeze(labels == i)
    #         X[:, mask_i] = 0
    #         X[i - 1, mask_i] = 1
    # else:
    #     X = _clean_labels_ar(X + 1, labels).reshape((5,5))
    #
    #
    # print "X"
    # print X
    # print X.shape

    # lap_sparse X = B

    # print "test:"
    #
    # l_x, l_y, l_z = tuple(data.shape[i] for i in range(3))
    # edges = _make_graph_edges_3d(l_x, l_y, l_z)
    # print edges
    #
    # weights = _compute_weights_3d(data, spacing, beta=130, eps=1.e-10,
    #                               multichannel=False)
    #
    # print weights
    # print len(weights)
    #
    # # if mask is not None:
    # #     edges, weights = _mask_edges_weights(edges, weights, mask)
    # lap = _make_laplacian_sparse(edges, weights)
    #
    # print lap
    #
    # print "_make_laplacian_sparse(...)"
    #
    # pixel_nb = edges.max() + 1
    # print pixel_nb
    #
    # diag = np.arange(pixel_nb)
    # print diag
    #
    # i_indices = np.hstack((edges[0], edges[1]))
    # j_indices = np.hstack((edges[1], edges[0]))
    # print i_indices
    # print j_indices
    #
    # lap_data = np.hstack((-weights, -weights))
    #
    # lap = sparse.coo_matrix((lap_data, (i_indices, j_indices)),
    #                         shape=(pixel_nb, pixel_nb))
    # print lap
    #
    # connect = - np.ravel(lap.sum(axis=1))
    # print connect
    #
    # lap = sparse.coo_matrix(
    #     (np.hstack((lap_data, connect)), (np.hstack((i_indices, diag)),
    #                                   np.hstack((j_indices, diag)))),
    #     shape=(pixel_nb, pixel_nb))
    #
    # print lap.tocsr()



    # del edges, weights
    # return lap

    # _build_laplacian(data, spacing)

    # from skimage.segmentation import random_walker
    # # from skimage.data import binary_blobs
    # import skimage
    #
    # # Generate noisy synthetic data
    # # data = skimage.img_as_float(binary_blobs(length=128, seed=1))
    # data = np.zeros((100, 100))
    # data[50:60, 50:70] = 1
    # data += 0.35 * np.random.randn(*data.shape)
    # markers = np.zeros(data.shape, dtype=np.uint)
    # markers[data < -0.3] = 1
    # markers[data > 1.3] = 2
    #
    # # Run random walker algorithm
    # labels = random_walker(data, markers, beta=10, mode='bf')
    #
    # # Plot results
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
    #                                     sharex=True, sharey=True)
    # ax1.imshow(data, cmap='gray', interpolation='nearest')
    # ax1.axis('off')
    # ax1.set_adjustable('box-forced')
    # ax1.set_title('Noisy data')
    # ax2.imshow(markers, cmap='hot', interpolation='nearest')
    # ax2.axis('off')
    # ax2.set_adjustable('box-forced')
    # ax2.set_title('Markers')
    # ax3.imshow(labels, cmap='gray', interpolation='nearest')
    # ax3.axis('off')
    # ax3.set_adjustable('box-forced')
    # ax3.set_title('Segmentation')
    #
    # fig.tight_layout()
    # plt.show()
