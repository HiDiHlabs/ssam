import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
import os
sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})
from functools import reduce
from sklearn.neighbors.kde import KernelDensity
from sklearn.preprocessing import normalize
from scipy import ndimage
#from scipy.stats import pearsonr
from tempfile import mkdtemp
from utils import corr, calc_ctmap

class SSAMDataset:
    def __init__(self, genes, positions, width, height, depth=1):
        """
        SSAMDataset(genes, positions, width, height, depth = 1, ncores = -1, save_dir = "", verbose = False)
            A class to store intial values and results of SSAM analysis.

            Parameters
            ----------
            genes : list(string)
                The genes that will be used for the analysis.
            positions : list(numpy.ndarray)
                Position of the mRNAs in um, given as a list of
                N x D ndarrays (N is number of mRNAs, D is number of dimensions).
            width : float
                Width of the image in um.
            height : float
                Height of the image in um.
            depth : float, optional
                Depth of the image in um. Depth == 1 means 2D image.
            
            Properties
            ----------
            vf : numpy.ndarray
                The vector field.
            vf_norm : numpy.ndarray
                L1 norm of the vector field.
        """
        if depth < 1 or width < 1 or height < 1:
            raise ValueError
        self.dataset.width = width
        self.dataset.height = height
        self.dataset.depth = depth
        self.dataset.genes = list(genes)
        self.positions = list(positions)
        self.dataset.is3d = depth > 1
        self.vf = None
        self.__vf_norm = None
        self.corr_map = None

        @property
        def vf_norm(self):
            if self.vf is None:
                return None
            if self.__vf_norm is None:
                self.__vf_norm = np.sum(self.vf, axis=len(self.vf.shape) - 1)
            return self.__vf_norm


class SSAMAnalysis:
    def __init__(self, dataset, ncores=-1, save_dir="", verbose=False):
        """
        SSAMAnalysis(ncores = -1, save_dir = "", verbose = False)
            A class to run SSAM analysis.

            Parameters
            ----------
            dataset : SSAMDataset
                A SSAMDataset object.
            ncores : int, optional
                Number of cores for parallel computation. If a negative value is given,
                (# of all available cores on system - value) cores will be used.
            save_dir : string, optional
                Directory to store intermediate data (e.g. density / vector field).
                Any data which already exists will be loaded and reused.
            verbose : bool, optional
                If True, then it prints out messages during the analysis.
        """
        self.dataset = dataset
        if not ncores > 0:
            ncores += multiprocessing.cpu_count()
        if ncores > multiprocessing.cpu_count():
            ncores = multiprocessing.cpu_count()
        if not ncores > 0:
            raise ValueError("Invalid number of cores.")
        self.ncores = ncores
        self.use_savedir = True
        if len(save_dir) == 0:
            save_dir = mkdtemp()
            self.use_savedir = False
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.verbose = verbose

    def __m__(self, message):
        if self.verbose:
            print(message)

    def run_kde(self, kernel="gaussian", bandwidth=2.0, sampling_distance=1.0, use_mmap=True):
        """
        run_kde(kernel = "gaussian", bandwidth = 2.0, sampling_distance = 1.0)
            Run KDE to estimate density of mRNA.

            Parameters
            ----------
            kernel : string, optional
                Kernel for density estimation.
            bandwidth : float, optional
                Parameter to adjust width of kernel.
                Set it 2 to make FWHM of Gaussian kernel to be ~10um (assume that avg. cell diameter is 10um).
            sampling_distance : float, optional
                Grid spacing in um.
            use_mmap : bool, optional
                Use MMAP to reduce memory usage during analysis.
                Turning off this option significantly increases analysis speed, but can use tremendous amount of memory.

            Returns
            -------
            out : numpy.ndarray
                The vector field.
        """
        steps = [
            int(np.ceil(self.dataset.width / sampling_distance)),
            int(np.ceil(self.dataset.height / sampling_distance)),
        ]
        if self.dataset.is3d:
            steps.append(
                int(np.ceil(self.dataset.depth / sampling_distance))
            )
        total_steps = np.prod(steps)
        vf_shape = tuple(steps + [len(self.dataset.genes), ])
        vf_filename = os.path.join(self.save_dir, 'vf')
        if (use_mmap and not os.path.exists(vf_filename + '.dat')) or \
                (not use_mmap and not os.path.exists(vf_filename + '.npy') and not os.path.exists(vf_filename + '.dat')):
            # If VF file doesn't exist, then run KDE
            if use_mmap:
                vf = np.memmap(vf_filename + '.dat.tmp', dtype='double', mode='w+', shape=vf_shape)
            else:
                vf = np.zeros(vf_shape)
            chunksize = min(np.ceil(total_steps / self.ncores), 100000)
            def yield_chunk():
                chunk = np.zeros(shape=[chunksize, len(steps)], dtype=int)
                cnt = 0
                remaining_cnt = total_steps
                for x in range(steps[0]):
                    for y in range(steps[1]):
                        if self.dataset.is3d:
                            for z in range(steps[2]):
                                chunk[cnt, :] = [x, y, z]
                                cnt += 1
                                if cnt == chunksize:
                                    yield chunk
                                    remaining_cnt -= cnt
                                    cnt = 0
                                    chunk = np.zeros(shape=[min(chunksize, remaining_cnt), len(steps)], dtype=int)
                        else:
                            chunk[cnt, :] = [x, y]
                            cnt += 1
                            if cnt == chunksize:
                                yield chunk
                                remaining_cnt -= cnt
                                cnt = 0
                                chunk = np.zeros(shape=[min(chunksize, remaining_cnt), len(steps)], dtype=int)
                if cnt > 0:
                    yield chunk

            def yield_chunks():
                chunks = []
                for chunk in yield_chunk():
                    chunks.append(chunk)
                    if len(chunks) == self.ncores:
                        yield chunks
                        chunks = []
                if len(chunks) > 0:
                    yield chunks

            pool = None
            for gidx, gene_name in enumerate(self.dataset.genes):
                pdf_filename = os.path.join(self.save_dir, 'pdf_sd%s_bw%s_%s.npy'%(
                    ('%f' % sampling_distance).rstrip('0').rstrip('.'),
                    ('%f' % bandwidth).rstrip('0').rstrip('.'),
                    gene_name)
                )
                if os.path.exists(pdf_filename):
                    self.__m__("Loading %s..."%gene_name)
                    pdf = np.load(pdf_filename)
                else:
                    self.__m__("Running KDE for %s..."%gene_name)
                    if pool is None:
                        pool = multiprocessing.Pool(self.ncores)
                    pdf = np.zeros(shape=vf_shape[:-1])
                    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(positions[gidx])
                    for chunks in yield_chunks():
                        pdf_chunks = pool.map(kde.score_samples, [chunk * sampling_distance for chunk in chunks])
                        for pdf_chunk, pos_chunk in zip(pdf_chunks, chunks):
                            if self.dataset.is3d:
                                pdf[pos_chunk[:, 0], pos_chunk[:, 1], pos_chunk[:, 2]] = np.exp(pdf_chunk)
                            else:
                                pdf[pos_chunk[:, 0], pos_chunk[:, 1]] = np.exp(pdf_chunk)
                    pdf /= np.sum(pdf)
                    np.save(pdf_filename, pdf)
                vf[..., gidx] = pdf * len(positions[gidx])
            if use_mmap:
                vf.flush()
                vf.close()
                os.rename(vf_filename + '.dat.tmp', vf_filename + '.dat')
                vf = np.memmap(vf_filename + '.dat', dtype='double', mode='r', shape=vf_shape)
            elif self.use_savedir:
                np.save(vf_filename + '.npy', vf)
        elif not use_mmap:
            if os.path.exists(vf_filename + '.npy'):
                vf = np.load(vf_filename + '.npy')
            else: # == os.path.exists(vf_filename + '.dat'):
                vf_tmp = np.memmap(vf_filename + '.dat', dtype='double', mode='r', shape=vf_shape)
                vf = np.array(vf_tmp, copy=True)
                if self.use_savedir:
                    np.save(vf_filename + '.npy', vf)
        elif use_mmap:
            vf = np.memmap(vf_filename + '.dat', dtype='double', mode='r', shape=vf_shape)
        self.dataset.vf = vf
        return

    def find_localmax(self, corr_size=3, search_size=20, min_norm=0):
        """
        find_localmax(search_size = 20, min_norm = 0)
            Find location of local maximum vectors in the correlation map of the vector field.

            Parameters
            ----------
            corr_size : int, optional
                Size of square (or cube) that is used to compute the local correlation values.
                This value should be an odd number.
            search_size : int, optional
                Size of square (or cube) that is used to search for the local maximum values.
                This value should be an odd number.
            min_norm : float, optional
                Minimum value of L1 norm of the local maximum.
            
            Returns
            -------
            out : self
                This method returns the current object, to make possible to chain methods.
                e.g. analysis.run_kde().find_localmax()
        """
        corr_half_size = int(corr_size / 2)
        if self.dataset.is_3d:
            corr_dx = np.meshgrid(range(corr_size), range(corr_size), range(corr_size))
        else:
            corr_dx = np.meshgrid(range(corr_size), range(corr_size))
        corr_dx = np.array(list(zip(*[np.ravel(e) - corr_half_size for e in corr_dx])))
        corr_pos = np.meshgrid(*[range(corr_half_size, e-corr_half_size) for e in self.dataset.vf_norm.shape])
        corr_map = np.zeros_like(self.dataset.vf_norm)
        self.dataset.corr_map = corr_map
        for x in zip(*[np.ravel(e) for e in corr_pos]):
            avg_corr = 0.0
            for dx in corr_dx:
                if all([e == 0 for e in dx]):
                    continue
                avg_corr += utils.corr(self.dataset.vf[x], self.dataset.vf[x + dx])
            corr_map[x] = avg_corr / len(corr_dx)
        max_mask = corr_map == ndimage.maximum_filter(corr_map, size=search_size)
        max_mask &= self.dataset.vf_norm > min_norm
        local_maxs = np.where(max_mask)
        self.__m__("Found %d local max vectors."%len(local_maxs[0]))
        self.dataset.local_maxs = local_maxs
        return

    def expand_localmax(self, r=0.95, min_pixels=7, max_pixels=1000):
        """
        expand_localmax(r = 0.95, p = 0.001, min_pixels = 7, max_pixels = 1000)
            Merge the vectors nearby the local max vectors.
            Only the vectors with the large Pearson correlation values are merged.

            Parameters
            ----------
            r : float, optional
                Minimum Pearson's correlation coefficient to look for the nearby vectors.
            min_pixels : float, optional
                Minimum number of pixels to merge.
            max_pixels : float, optional
                Maximum number of pixels to merge.
            
            Returns
            -------
            out : self
                This method returns the current object, to make possible to chain methods.
                e.g. analysis.find_localmax().expand_localmax()
        """
        avg_vecs = []
        self.__m__("Expanding local max vectors...")
        if self.dataset.is_3d:
            fill_dx = np.meshgrid(range(3), range(3), range(3))
        else:
            fill_dx = np.meshgrid(range(3), range(3))
        fill_dx = np.array(list(zip(*[np.ravel(e) - 1 for e in fill_dx])))
        mask = np.zeros(self.dataset.vf.shape[:-1]) # TODO: sparse?
        def flood_fill(pos):
            refvec = self.dataset.vf[pos]
            filled_poslist = []
            def inner_flood_fill(_pos):
                if len(filled_poslist) > max_pixels:
                    return
                if any([p < 0 or p >= self.dataset.vf.shape[i] for i, p in enumerate(_pos)]) or _pos in filled_poslist:
                    return
                if corr(self.dataset.vf[_pos], refvec) > r:
                    filled_poslist.append(_pos)
                else:
                    return
                for dx in fill_dx:
                    if all([e == 0 for e in dx]):
                        continue
                    inner_flood_fill(tuple(np.array(_pos) + dx))
            inner_flood_fill(pos)
            if len(filled_poslist) < min_pixels or len(filled_poslist) > max_pixels:
                return ()
            else:
                return filled_poslist

        nlocalmaxs = len(self.dataset.local_maxs[0])
        for cnt, idx in enumerate(range(nlocalmaxs), start=1):
            filled_pos = tuple(zip(*flood_fill(tuple(i[idx] for i in self.dataset.local_maxs))))
            mask[filled_pos] = 1
            if len(filled_pos) > 0:
                avg_vecs.append(np.mean(self.dataset.vf[filled_pos], axis=0))
            if cnt % 100 == 0:
                self.__m__("Processed %d/%d..."%(cnt, nlocalmaxs))
        self.__m__("Processed %d/%d..."%(cnt, nlocalmaxs))
        self.dataset.expanded_vectors = np.array(avg_vecs)
        self.dataset.expanded_mask = mask
        return

    def cluster_vectors(self, method="louvain", **kwargs):
        """
        cluster_vectors(method = "louvain", **kwargs)
            Cluster the given vectors using the specified clustering method.

            Parameters
            ----------
            vecs : numpy.ndarray
                Vectors to cluster, which is 2D N x D matrix (N: number of vectors, D: number of genes).
            method : string
                Clustring method. Possible values: "louvain", "kmeans"

            Other Parameters
            ----------
            resolution: float, optional (default: 0.6)
                Resolution for Louvain community detection.
            min_cluster_size: int, optional (default: 10)
                Set minimum cluster size from Louvain community detection.
            k : int, optional (default: 50)
                Initial k value for Kmeans clustering.
            r : float, optional (default: 0.95)
                If r < 1.0, then the clusters with a Pearson's correlation coefficienct larger than r are merged.

            Returns
            -------
            out : numpy.ndarray
                Numpy array containing centroids.
                A centroid of a cluster is calculated by taking average of all vectors in the cluster.
        """
        vecs_normalized = normalize(self.dataset.expanded_vectors, "l1")

        if method == "louvain":
            from sklearn.neighbors import kneighbors_graph
            import community
            import networkx as nx
            louvain_resolution = kwargs.get("resolution", 0.6)
            louvain_resolution = kwargs.get("min_cluster_size", 10)
            knn_graph = kneighbors_graph(vecs_normalized, 20, mode='connectivity', include_self=False)
            snn_graph = (knn_graph + knn_graph.T == 2).astype(int)
            G = nx.from_scipy_sparse_matrix(snn_graph)
            partition = community.best_partition(G, resolution=louvain_resolution)
            lbls = np.array(list(partition.values()))
            low_clusters = []
            for lbl in set(lbls):
                cnt = np.sum(lbls == lbl)
                if cnt < min_cluster_size:
                    low_clusters.append(lbl)
            for lbl in low_clusters:
                lbls[lbls == lbl] = -1
        elif method == "kmeans":
            from sklearn.cluster import KMeans
            kmeans_k = kwargs.get("k", 50)
            merge_r = kwargs.get("r", 0.95)
            km = KMeans(n_clusters=kmeans_k).fit(vecs_normalized)
            lbls = km.labels_[:]

            centroids = []
            for lbl in set(km.labels_):
                centroids.append(np.mean(vecs_normalized[km.labels_ == lbl, :], axis=0))

            if merge_r < 1.0:
                merge_ref = []
                for i in range(len(centroids)):
                    for j in range(i+1, len(centroids)):
                        pr = corr(centroids[i], centroids[j])
                        if pr > merge_r:
                            appended = False
                            for mr in merge_ref:
                                if i in mr:
                                    mr.append(j)
                                    appended = True
                                elif j in mr:
                                    mr.append(i)
                                    appended = True
                                if appended == True:
                                    break
                            if not appended:
                                merge_ref.append([i, j])

                for mr in merge_ref:
                    for cidx in mr[1:]:
                        lbls[lbls == cidx] = mr[0]
        else:
            raise ValueError("This method is not supported.")
        
        centroids = []
        for lbl in set(lbls):
            centroids.append(np.mean(vecs_normalized[lbls == lbl, :], axis=0))

        self.dataset.centroids = np.array(centroids)
        return

    def map_celltypes(self, min_r = 0.6):
        """
        map_celltypes()
            Create correlation maps between the centroids and the vector field.
            Each correlation map corresponds each cell type's image map.

            Parameters
            ----------
            min_r : float
                Threshold for mimimum Pearson's correlation coefficient between the centroids and the vector field.

            Returns
            -------
            out : numpy.ndarray
                Returns cell type map.
        """
        from scipy import sparse
        celltype_maps = []
        for centroid in self.dataset.centroids:
            ctmap = calc_ctmap(centroid, self.dataset.vf, self.ncores)
            ctmap[ctmap < min_r] = 0
            celltype_maps.append(sparse.csr_matrix(ctmap))
        self.dataset.celltype_maps = celltype_maps
        return

    def run_full_analysis(self):
        """
        run_full_analysis()
            Run all analyses with default parameters.

            Parameters
            ----------
            pos_dic : dict(string: numpy.ndarray)
                Position of the mRNAs in um, given as a dictionary.
                Value is ndarray, (# of rows) = (number of mRNAs), (# of cols) = (# of dimension). Key is the gene name.

            Returns
            -------
            out : list
                List of the cell type maps.
        """
        self.run_kde()
        self.find_localmax()
        self.expand_localmax()
        self.cluster_vectors()
        self.map_celltypes()
        return

if __name__ == "__main__":
    import pickle
    with open("/media/pjb7687/data/ssam_test/test.pickle", "rb") as f:
        selected_genes, pos_dic, width, height, depth = pickle.load(f)
    
    dataset = SSAMDataset(selected_genes, [pos_dic[gene] for gene in selected_genes] width, height, depth)
    analysis = SSAMAnalysis(dataset, save_dir="/media/pjb7687/data/ssam_test2", verbose=True)
    analysis.run_kde()
    analysis.find_localmax()
    analysis.expand_localmax()