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
import scipy
from scipy import ndimage
from sklearn.decomposition import PCA
#from scipy.stats import pearsonr
from tempfile import mkdtemp
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
import community
import networkx as nx
import sparse
from skimage import filters
from skimage.morphology import disk
from skimage import measure
from .utils import corr, calc_ctmap, calc_corrmap, flood_fill

class SSAMDataset(object):
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
        self.width = width
        self.height = height
        self.depth = depth
        self.genes = list(genes)
        self.positions = list(positions)
        self.is3d = depth > 1
        self.__vf = None
        self.__vf_norm = None
        self.normalized_vectors = None
        self.expanded_vectors = None
        #self.corr_map = None
        self.tsne = None

    @property
    def vf(self):
        return self.__vf
    
    @vf.setter
    def vf(self, vf):
        self.__vf = vf
        self.__vf_norm = None    
        
    @property
    def vf_norm(self):
        if self.vf is None:
            return None
        if self.__vf_norm is None:
            self.__vf_norm = np.sum(self.vf, axis=len(self.vf.shape) - 1)
        return self.__vf_norm
    
    def plot_tsne(self, run_tsne=True, n_iter=5000, perplexity=70, early_exaggeration=10, metric="correlation", cmap="jet"):
        if run_tsne:
            self.tsne = TSNE(n_iter=n_iter, perplexity=perplexity, early_exaggeration=early_exaggeration, metric=metric).fit_transform(self.normalized_vectors[self.cluster_labels != -1, :])
        plt.scatter(self.tsne[:, 0], self.tsne[:, 1], c=self.cluster_labels[self.cluster_labels != -1], cmap=cmap)
        return
    
    def plot_expanded_mask(self):
        plt.imshow(self.expanded_mask, vmin=0, vmax=1)
        return
    
    def plot_correlation_map(self):
        plt.imshow(ds.corr_map, vmin=0.995, vmax=1.0)
        plt.colorbar()
        return

class SSAMAnalysis(object):
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
        vf_filename = os.path.join(self.save_dir, 'vf_sd%s_bw%s'%(
            ('%f' % sampling_distance).rstrip('0').rstrip('.'),
            ('%f' % bandwidth).rstrip('0').rstrip('.')
        ))
        if (use_mmap and not os.path.exists(vf_filename + '.dat')) or \
                (not use_mmap and not os.path.exists(vf_filename + '.npy') and not os.path.exists(vf_filename + '.dat')):
            # If VF file doesn't exist, then run KDE
            if use_mmap:
                vf = np.memmap(vf_filename + '.dat.tmp', dtype='double', mode='w+', shape=vf_shape)
            else:
                vf = np.zeros(vf_shape)
            chunksize = min(int(np.ceil(total_steps / self.ncores)), 100000)
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
                    pdf = np.zeros(shape=vf_shape[:-1])
                    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(self.dataset.positions[gidx])
                    if pool is None:
                        pool = multiprocessing.Pool(self.ncores)
                    for chunks in yield_chunks():
                        pdf_chunks = pool.map(kde.score_samples, [chunk * sampling_distance for chunk in chunks])
                        for pdf_chunk, pos_chunk in zip(pdf_chunks, chunks):
                            if self.dataset.is3d:
                                pdf[pos_chunk[:, 0], pos_chunk[:, 1], pos_chunk[:, 2]] = np.exp(pdf_chunk)
                            else:
                                pdf[pos_chunk[:, 0], pos_chunk[:, 1]] = np.exp(pdf_chunk)
                    pdf /= np.sum(pdf)
                    np.save(pdf_filename, pdf)
                vf[..., gidx] = pdf * len(self.dataset.positions[gidx])
            if use_mmap:
                vf.flush()
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

    def calc_correlation_map(self, corr_size=3):
        """
        calc_correlation_map(corr_size = 3)
            Calculate local correlation map of the vector field.

            Parameters
            ----------
            corr_size : int, optional
                Size of square (or cube) that is used to compute the local correlation values.
                This value should be an odd number.
        """
        
        corr_map = calc_corrmap(self.dataset.vf, ncores=self.ncores, size=int(corr_size/2))
        self.dataset.corr_map = np.array(corr_map, copy=True)
        return
    
    def find_localmax(self, search_size=20, min_norm=0):
        """
        find_localmax(search_size = 20, min_norm = 0)
            Find location of local maximum vectors in the L1 norm of the vector field.

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
        """
        
        #corr_map = calc_corrmap(self.dataset.vf, ncores=self.ncores, size=int(corr_size/2))
        #self.dataset.corr_map = np.array(corr_map, copy=True)
        #if self.dataset.is3d and corr_map.shape[-1] < 5:
        #    selected_z = int(corr_map.shape[-1]/2)
        #    self.__m__("Warning: depth of image is very small, only z == %d is considered for finding local maximum vectors."%selected_z)
        #    corr_map = corr_map[..., selected_z]
        #    max_mask = corr_map == ndimage.maximum_filter(corr_map, size=search_size)
        #    tmp = np.zeros(self.dataset.vf_norm.shape, dtype=bool)
        #    tmp[..., selected_z] = max_mask
        #    max_mask = tmp
        #else:
        #    max_mask = corr_map == ndimage.maximum_filter(corr_map, size=search_size)
        max_mask = self.dataset.vf_norm == ndimage.maximum_filter(self.dataset.vf_norm, size=search_size)
        max_mask &= self.dataset.vf_norm > min_norm
        local_maxs = np.where(max_mask)
        self.__m__("Found %d local max vectors."%len(local_maxs[0]))
        self.dataset.local_maxs = local_maxs
        return

    def expand_localmax(self, r=0.99, min_pixels=7, max_pixels=1000):
        """
        expand_localmax(r = 0.99, p = 0.001, min_pixels = 7, max_pixels = 1000)
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
        """
        
        expanded_vecs = []
        self.__m__("Expanding local max vectors...")
        if self.dataset.is3d:
            fill_dx = np.meshgrid(range(3), range(3), range(3))
        else:
            fill_dx = np.meshgrid(range(3), range(3))
        fill_dx = np.array(list(zip(*[np.ravel(e) - 1 for e in fill_dx])))
        mask = np.zeros(self.dataset.vf.shape[:-1]) # TODO: sparse?
        nlocalmaxs = len(self.dataset.local_maxs[0])
        valid_pos_list = []
        for cnt, idx in enumerate(range(nlocalmaxs), start=1):
            local_pos = tuple(i[idx] for i in self.dataset.local_maxs)
            filled_pos = tuple(zip(*flood_fill(local_pos, self.dataset.vf, r, min_pixels, max_pixels)))
            if len(filled_pos) > 0:
                mask[filled_pos] = 1
                valid_pos_list.append(local_pos)
                expanded_vecs.append(np.sum(self.dataset.vf[filled_pos], axis=0))
            if cnt % 100 == 0:
                self.__m__("Processed %d/%d..."%(cnt, nlocalmaxs))
        self.__m__("Processed %d/%d..."%(cnt, nlocalmaxs))
        self.dataset.expanded_vectors = np.array(expanded_vecs)
        self.dataset.expanded_mask = mask
        self.dataset.valid_local_maxs = valid_pos_list
        return
    
    def normalize_vectors(self, norm="l1", use_expanded_vectors=False, normalize_gene=True, normalize_vector=True):
        """
        nomalize_vectors(norm = "l1", use_expanded_vectors = False, normalize_gene = True, normalize_vector = True)
            Normalize vectors using scipy.preprocessing.normalization
            
            Parameters
            ----------
            norm : string (default: "l1")
                Type of norm for normalization. 
            use_expanded_vectors : bool (default: False)
                If True, use averaged vectors nearby local maxima of the vector field.
            normalize_gene: bool (default: True)
                If True, normalize vectors by sum of each gene expression across all vectors.
            normalize_vector: bool (default: True)
                If True, normalize vectors by sum of all gene expression of each vector.
        """
        if use_expanded_vectors:
            vec = np.array(self.dataset.expanded_vectors, copy=True)
        else:
            vec = np.array(self.dataset.vf[self.dataset.local_maxs], copy=True)
        if normalize_gene:
            vec = normalize(vec, norm=norm, axis=0) * vec.shape[1] # Normalize per gene
        if normalize_vector:
            vec = normalize(vec, norm=norm, axis=1) * vec.shape[0] # Normalize per vector
        self.dataset.normalized_vectors = vec
        return
    
    def cluster_vectors(self, pca_dims=10, min_cluster_size=0, resolution=0.8, prune=1.0/15.0, snn_neighbors=30, metric="euclidean", subclustering=False, seed=-1, log_transform=False):
        """
        cluster_vectors(method = "louvain", **kwargs)
            Cluster the given vectors using the specified clustering method.

            Parameters
            ----------
            pca_dims : int (default: 10)
                Number of principal componants used for clustering.
            min_cluster_size: int, optional (default: 0)
                Set minimum cluster size.
            resolution: float, optional (default: 1.0)
                Resolution for Louvain community detection.
            prune: float, optional (default: 1.0/15.0)
                Threshold for Jaccard index (weight of SNN network). If it is smaller than prune, it is set to zero.
            snn_neighbors: int, optional (default: 30)
                Number of neighbors for SNN network.
            metric: string, optional (default: "euclidean")
                Metric for calculation of distance between vectors in gene expression space.
            subclustering: bool, optional (default: False)
                If True, each cluster will be clustered once again with the same method to find more subclusters.
            seed: int, optional (default: -1)
                Random seed for replication (TODO: at the moment it is not working).
            log_transform: bool, optional (default: False)
                if True, clustering will be done after taking log(x+1) of the vectors. The cluster centroids and medoids will be restored back with exp(x)-1.
        """
        
        vecs_normalized = self.dataset.normalized_vectors
        if log_transform:
            vecs_normalized = np.log1p(vecs_normalized)
        vecs_normalized_dimreduced = PCA(n_components=pca_dims).fit_transform(vecs_normalized)

        def cluster_vecs(vecs):
            k = min(snn_neighbors, vecs.shape[0])
            knn_graph = kneighbors_graph(vecs, k, mode='connectivity', include_self=True, metric=metric).todense()
            #snn_graph = (knn_graph + knn_graph.T == 2).astype(int)
            intersections = np.dot(knn_graph, knn_graph.T)
            #unions = np.zeros_like(intersections)
            #for i in range(knn_graph.shape[0]):
            #    for j in range(knn_graph.shape[1]):
            #        unions[i, j] = np.sum(np.logical_or(knn_graph[i, :], knn_graph[j, :]))
            #snn_graph = intersections / unions
            snn_graph = intersections / (k + (k - intersections)) # borrowed from Seurat
            snn_graph[snn_graph < prune] = 0
            G = nx.from_numpy_matrix(snn_graph)
            partition = community.best_partition(G, resolution=resolution, randomize=True)
            lbls = np.array(list(partition.values()))
            low_clusters = []
            cluster_indices = []
            for lbl in set(list(lbls)):
                cnt = np.sum(lbls == lbl)
                if cnt < min_cluster_size:
                    low_clusters.append(lbl)
                else:
                    cluster_indices.append(lbls == lbl)
            for lbl in low_clusters:
                lbls[lbls == lbl] = -1
            for i, idx in enumerate(cluster_indices):
                lbls[idx] = i
            return lbls
        
        if subclustering:
            super_lbls = cluster_vecs(vecs_normalized_dimreduced)
            all_lbls = np.zeros_like(super_lbls)
            global_lbl_idx = 0
            for super_lbl in set(list(super_lbls)):
                super_lbl_idx = np.where(super_lbls == super_lbl)[0]
                if super_lbl == -1:
                    all_lbls[super_lbl_idx] = -1
                    continue
                sub_lbls = cluster_vecs(vecs_normalized_dimreduced[super_lbl_idx])
                for sub_lbl in set(list(sub_lbls)):
                    if sub_lbl == -1:
                        all_lbls[[super_lbl_idx[sub_lbls == sub_lbl]]] = -1
                        continue
                    all_lbls[[super_lbl_idx[sub_lbls == sub_lbl]]] = global_lbl_idx
                    global_lbl_idx += 1
        else:
            all_lbls = cluster_vecs(vecs_normalized_dimreduced)
        
        centroids = []
        medoids = []
        for lbl in sorted(list(set(all_lbls))):
            if lbl == -1:
                continue
            cl_vecs = vecs_normalized[all_lbls == lbl, :]
            if log_transform:
                cl_vecs = np.log(cl_vecs + 1)
            cl_dists = scipy.spatial.distance.cdist(cl_vecs, cl_vecs, metric)
            medoid = cl_vecs[np.argmin(np.sum(cl_dists, axis=0))]
            centroid = np.mean(cl_vecs, axis=0)
            if log_transform:
                medoid = np.exp(medoid) - 1
                centroid = np.exp(centroid) - 1
            medoids.append(medoid)
            centroids.append(centroid)
            
        self.dataset.centroids = np.array(centroids)
        self.dataset.medoids = np.array(medoids)
        self.dataset.cluster_labels = all_lbls
        return

    def map_celltypes(self, centroids=None, min_r=0.6, min_norm="otsu", filter_params={}, min_blob_area=75, log_transform=False):
        """
        map_celltypes(centroids = None, min_r = 0.6, log_transform = False)
            Create correlation maps between the centroids and the vector field.
            Each correlation map corresponds each cell type's image map.

            Parameters
            ----------
            centroids: np.array(float) or list(np.array(float)), default: None
                If given, map celltypes with the given cluster centroids.
                Otherwise, map celltypes with the clusters from the vector field.
            min_r: float, default: 0.6
                Threshold for mimimum Pearson's correlation coefficient between the centroids and the vector field.
            log_transform: bool, default: False
                If True, the cluster centroids are mapped to vector field after taking log(x+1).
        """
        
        celltype_maps = []
        if centroids is None:
            centroids = self.dataset.centroids
        for centroid in centroids:
            if log_transform:
                ctmap = calc_ctmap(np.log(centroid + 1), np.log(self.dataset.vf + 1), self.ncores)
            else:
                ctmap = calc_ctmap(centroid, self.dataset.vf, self.ncores)
            ctmap[ctmap < min_r] = 0
            
            if isinstance(min_norm, str):
                if min_norm in ["local", "niblack", "sauvola", "localotsu"]:
                    im = np.zeros_like(self.dataset.vf_norm)
                    im[ctmap > min_r] = self.dataset.vf_norm[ctmap > min_r]
                if min_norm == "localotsu":
                    max_norm = np.max(im)
                    im /= max_norm
                    selem = disk(filter_params['radius'])
                    min_norm_cut = filters.rank.otsu(im, selem) * max_norm
                else:
                    filter_func = getattr(filters, "threshold_" + min_norm)
                    if min_norm in ["local", "niblack", "sauvola"]:
                        min_norm_cut = filter_func(im, **filter_params)
                    else:
                        highr_norm = self.dataset.vf_norm[ctmap > min_r]
                        #sigma = np.std(highr_norm)
                        min_norm_cut = filter_func(highr_norm, **filter_params)
            else:
                min_norm_cut = float(min_norm)
            ctmap[self.dataset.vf_norm < min_norm_cut] = 0
            if min_blob_area > 0:
                mask = ctmap > 0
                blob_labels = measure.label(mask, background=0)
                for bp in measure.regionprops(blob_labels):
                    minr, minc, maxr, maxc = bp.bbox
                    if bp.filled_area < min_blob_area:
                        mask[minr:maxr, minc:maxc] = 0
                        continue
                    if bp.area != bp.filled_area:
                        mask[minr:maxr, minc:maxc] = bp.filled_image
            ctmap[~mask] = 0
            celltype_maps.append(sparse.COO(np.where(ctmap > 0), ctmap[ctmap > 0], shape=ctmap.shape))
        self.dataset.celltype_maps = celltype_maps
        return

    def run_full_analysis_with_defaults(self):
        """
        run_full_analysis_with_defaults()
            Run all analyses with the default parameters.
        """
        
        self.run_kde()
        self.find_localmax()
        self.normalize_vectors()
        self.cluster_vectors()
        self.map_celltypes()
        return

if __name__ == "__main__":
    import pickle
    with open("/media/pjb7687/data/ssam_test/test.pickle", "rb") as f:
        selected_genes, pos_dic, width, height, depth = pickle.load(f)
    
    dataset = SSAMDataset(selected_genes, [pos_dic[gene] for gene in selected_genes], width, height, depth)
    analysis = SSAMAnalysis(dataset, save_dir="/media/pjb7687/data/ssam_test2", verbose=True)
    analysis.run_kde()
    analysis.find_localmax()
    analysis.expand_localmax()
