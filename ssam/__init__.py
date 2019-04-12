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
from sklearn import preprocessing
import scipy
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
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
from matplotlib.colors import ListedColormap
import pickle
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
        self.cluster_labels = None
        #self.corr_map = None
        self.tsne = None
        self.umap = None

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
    
    def plot_l1norm(self, cmap="viridis", rotate=0):
        if rotate < 0 or rotate > 3:
            raise ValueError("rotate can only be 0, 1, 2, 3")
        im = np.array(self.vf_norm, copy=True)
        if rotate == 1 or rotate == 3:
            im = im.T
        if len(self.vf_norm.shape) == 3:
            plt.imshow(im[int(self.vf_norm.shape[2] / 2)], cmap=cmap)
        else:
            plt.imshow(im, cmap=cmap)
        if rotate == 1:
            plt.gca().invert_xaxis()
        elif rotate == 2:
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()
        elif rotate == 3:
            plt.gca().invert_yaxis()

    def plot_localmax(self, c=None, s=1, rotate=0):
        if rotate < 0 or rotate > 3:
            raise ValueError("rotate can only be 0, 1, 2, 3")
        if rotate == 0 or rotate == 2:
            dim0, dim1 = 1, 0
        elif rotate == 1 or rotate == 3:
            dim0, dim1 = 0, 1
        plt.scatter(self.local_maxs[dim0], self.local_maxs[dim1], s=s, c=None)
        plt.xlim([0, self.vf_norm.shape[dim0]])
        plt.ylim([self.vf_norm.shape[dim1], 0])
        if rotate == 1:
            plt.gca().invert_xaxis()
        elif rotate == 2:
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()
        elif rotate == 3:
            plt.gca().invert_yaxis()    
        
    def __run_pca(self, exclude_bad_clusters, pca_dims, log_transform):
        if exclude_bad_clusters:
            good_vecs = self.normalized_vectors[self.cluster_labels != -1, :]
        else:
            good_vecs = self.normalized_vectors
        if log_transform:
            X = np.log(good_vecs + 1)
        else:
            X = good_vecs
        return PCA(n_components=pca_dims).fit_transform(X)
        
    def plot_tsne(self, run_tsne=False, pca_dims=10, n_iter=5000, perplexity=70, early_exaggeration=10, metric="correlation", exclude_bad_clusters=True, log_transform=False, s=None, random_state=0, colors=[], cmap="jet"):
        if run_tsne or self.tsne is None:
            pcs = self.__run_pca(exclude_bad_clusters, pca_dims, log_transform)
            self.tsne = TSNE(n_iter=n_iter, perplexity=perplexity, early_exaggeration=early_exaggeration, metric=metric, random_state=random_state).fit_transform(pcs[:, :pca_dims])
        if self.cluster_labels is not None:
            if exclude_bad_clusters:
                cols = self.cluster_labels[self.cluster_labels != -1]
            else:
                cols = self.cluster_labels
        else:
            cols = None
        if len(colors) > 0:
            cmap = ListedColormap(colors)
        plt.scatter(self.tsne[:, 0], self.tsne[:, 1], s=s, c=cols, cmap=cmap)
        return

    def plot_umap(self, run_umap=False, pca_dims=10, metric="correlation", exclude_bad_clusters=True, log_transform=False, s=None, random_state=0, colors=[], cmap="jet"):
        if run_umap or self.umap is None:
            pcs = self.__run_pca(exclude_bad_clusters, pca_dims, log_transform)
            self.umap = UMAP(metric=metric, random_state=random_state).fit_transform(pcs[:, :pca_dims])
        if self.cluster_labels is not None:
            if exclude_bad_clusters:
                cols = self.cluster_labels[self.cluster_labels != -1]
            else:
                cols = self.cluster_labels
        else:
            cols = None
        if len(colors) > 0:
            cmap = ListedColormap(colors)
        plt.scatter(self.umap[:, 0], self.umap[:, 1], s=s, c=cols, cmap=cmap)
        return

    def plot_expanded_mask(self, cmap='Greys'):
        plt.imshow(self.expanded_mask, vmin=0, vmax=1, cmap=cmap)
        return
    
    def plot_correlation_map(self, cmap='hot'):
        plt.imshow(ds.corr_map, vmin=0.995, vmax=1.0, cmap=cmap)
        plt.colorbar()
        return
    
    def plot_celltypes_map(self, output_mask=None, background="black", centroid_indices=[], colors=None, min_r=0.6, min_norm=0.1, rotate=0):
        if len(centroid_indices) == 0:
            centroid_indices = range(len(self.celltype_maps))

        if colors is None:
            cmap = plt.get_cmap('jet')
            colors = cmap([float(i) / (len(centroid_indices) - 1) for i in range(len(centroid_indices))])
            
        all_colors = ["#000000" if not j in centroid_indices else colors[i] for i, j in enumerate(range(len(self.celltype_maps)))]
        cmap = ListedColormap(all_colors)
        if rotate == 1 or rotate == 3:
            max_corr_idx = self.max_corr_idx.T
            vf_norm = self.vf_norm.T
            max_corr = self.max_corr.T
        else:
            max_corr_idx = self.max_corr_idx
            vf_norm = self.vf_norm
            max_corr = self.max_corr

        sctmap = cmap(max_corr_idx)
        if output_mask is not None:
            sctmap[~output_mask.astype(bool)] = (0, 0, 0, 0)
        sctmap[vf_norm < min_norm] = (0, 0, 0, 0)
        sctmap[max_corr < min_r] = (0, 0, 0, 0)
        
        #alpha = max_corr - fade_start
        #alpha[alpha < 0] = 0
        #alpha /= np.max(alpha)
        #sctmap[..., 3] = alpha
        
        plt.gca().set_facecolor(background)
        plt.imshow(sctmap)
        
        if rotate == 1:
            plt.gca().invert_xaxis()
        elif rotate == 2:
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()
        elif rotate == 3:
            plt.gca().invert_yaxis()

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
        def save_pickle(fn, o):
            with open(fn, "wb") as f:
                return pickle.dump(o, f, protocol=4)
        def load_pickle(fn):
            with open(fn, "rb") as f:
                return pickle.load(f)
        
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
                (not use_mmap and not os.path.exists(vf_filename + '.pkl') and not os.path.exists(vf_filename + '.dat')):
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
                save_pickle(vf_filename + '.pkl', vf)
        elif not use_mmap:
            if os.path.exists(vf_filename + '.pkl'):
                vf = load_pickle(vf_filename + '.pkl')
            else: # == os.path.exists(vf_filename + '.dat'):
                vf_tmp = np.memmap(vf_filename + '.dat', dtype='double', mode='r', shape=vf_shape)
                vf = np.array(vf_tmp, copy=True)
                if self.use_savedir:
                    save_pickle(vf_filename + '.pkl', vf)
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
    
    def find_localmax(self, search_size=21, min_norm=0, mask=None):
        """
        find_localmax(search_size = 21, min_norm = 0)
            Find local maxima vectors in the norm of the vector field.

            Parameters
            ----------
            search_size : int, optional
                Size of square (or cube) that is used to search for the local maximum values.
                This value should be an odd number.
            min_norm : float, optional
                Minimum value of norm at the local maxima.
            mask: numpy.ndarray, optional
                If given, find vectors in the masked region, instead of the whole image.
        """

        max_mask = self.dataset.vf_norm == ndimage.maximum_filter(self.dataset.vf_norm, size=search_size)
        max_mask &= self.dataset.vf_norm > min_norm
        if mask is not None:
            max_mask &= mask
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
    
    def normalize_vectors(self, norm="l1", use_expanded_vectors=False, normalize_gene=False, normalize_vector=True, log_transform=True, scale=True):
        """
        nomalize_vectors(norm = "l1", use_expanded_vectors = False, normalize_gene = False, normalize_vector = True, log_transform = True, scale = True)
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
            log_transform: bool (default: True)
                If True, vectors are log transformed.
            scale: bool (default: True)
                if True, vectors are z-scaled (mean centered and scaled by stdev).
        """
        if use_expanded_vectors:
            vec = np.array(self.dataset.expanded_vectors, copy=True)
        else:
            vec = np.array(self.dataset.vf[self.dataset.local_maxs], copy=True)
        if normalize_gene:
            vec = preprocessing.normalize(vec, norm=norm, axis=0) # Normalize per gene
        if normalize_vector:
            vec = preprocessing.normalize(vec, norm=norm, axis=1) # Normalize per vector
        if log_transform:
            vec = np.log2(vec + 1)
        if scale:
            ds.gene_means = np.mean(ds.normalized_vectors, axis=0)
            ds.gene_stds = np.std(ds.normalized_vectors, axis=0)
            vec = (vec - ds.gene_means) / ds.gene_stds
            #vec = preprocessing.scale(vec)
        self.dataset.normalized_vectors = vec
        return
    
    def cluster_vectors(self, pca_dims=10, min_cluster_size=0, resolution=0.8, prune=1.0/15.0, snn_neighbors=30, metric="euclidean", subclustering=False, random_state=0):
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
            random_state: int or random state object, optional (default: 0)
                Random seed or scikit-learn's random state object to replicate result
        """
        
        vecs_normalized = self.dataset.normalized_vectors
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
            partition = community.best_partition(G, resolution=resolution, random_state=random_state)
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
            cl_dists = scipy.spatial.distance.cdist(cl_vecs, cl_vecs, metric)
            medoid = cl_vecs[np.argmin(np.sum(cl_dists, axis=0))]
            centroid = np.mean(cl_vecs, axis=0)
            medoids.append(medoid)
            centroids.append(centroid)
            
        self.dataset.centroids = np.array(centroids)
        self.dataset.medoids = np.array(medoids)
        self.dataset.cluster_labels = all_lbls
        return

    def map_celltypes(self, centroids=None, use_medoids=False, min_r=0.6, min_norm="otsu", filter_params={}, min_blob_area=75, log_transform=True):
        """
        map_celltypes(centroids = None, min_r = 0.6, log_transform = False)
            Create correlation maps between the centroids and the vector field.
            Each correlation map corresponds each cell type's image map.

            Parameters
            ----------
            centroids: np.array(float) or list(np.array(float)), default: None
                If given, map celltypes with the given cluster centroids. Ignore 'use_medoids' parameter.
            use_medoids: bool, default: False
                Map cluster medoids instead of centroids.
            min_r: float, default: 0.6
                Threshold for mimimum Pearson's correlation coefficient between the centroids and the vector field.
            log_transform: bool, default: True
                Log transform vector field before calculating correlation.
        """
        
        celltype_maps = []
        if centroids is None:
            centroids = self.dataset.centroids
        if isinstance(min_norm, str):
            filter_offset = filter_params.pop('offset', 0)
        for centroid in centroids:
            ctmap = calc_ctmap(centroid, self.dataset.vf, self.ncores, log_transform)
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
                        if len(highr_norm) == 0 or np.max(highr_norm) == np.min(highr_norm):
                            min_norm_cut = np.max(vf_norm)
                        else:
                            min_norm_cut = filter_func(highr_norm, **filter_params)
                min_norm_cut -= filter_offset
            else:
                min_norm_cut = min_norm
            ctmap[self.dataset.vf_norm < min_norm_cut] = 0
            if min_blob_area > 0:
                mask = ctmap > 0
                blob_labels = measure.label(mask, background=0)
                for bp in measure.regionprops(blob_labels):
                    if len(bp.bbox) == 4:
                        minr, minc, maxr, maxc = bp.bbox
                        if bp.filled_area < min_blob_area:
                            for c in bp.coords:
                                mask[c[0], c[1]] = 0
                            continue
                        if bp.area != bp.filled_area:
                            mask[minr:maxr, minc:maxc] |= bp.filled_image
                    else:
                        minr, minc, minz, maxr, maxc, maxz = bp.bbox
                        if bp.filled_area < min_blob_area:
                            for c in bp.coords:
                                mask[c[0], c[1], c[2]] = 0
                            continue
                        if bp.area != bp.filled_area:
                            mask[minr:maxr, minc:maxc, minz:maxz] |= bp.filled_image
                ctmap[~mask] = 0
            celltype_maps.append(sparse.COO(np.where(ctmap > 0), ctmap[ctmap > 0], shape=ctmap.shape))
        
        newaxis = 3 if self.dataset.is3d else 2
        stacked_ctmaps = sparse.stack(celltype_maps, axis=newaxis)
        #self.dataset.max_corr = np.zeros_like(self.dataset.vf_norm)
        #self.dataset.max_corr_idx = np.zeros_like(self.dataset.vf_norm, dtype='int')
        #if newaxis == 2:
        #    for i in range(stacked_ctmaps.shape[0]):
        #        for j in range(stacked_ctmaps.shape[1]):
        #            corr_vec = stacked_ctmaps[i, j, :].todense()
        #            max_corr_vec_idx = np.argmax(corr_vec)
        #            max_corr_vec = corr_vec[max_corr_vec_idx]
        #            self.dataset.max_corr_idx[i, j] = max_corr_vec_idx
        #            self.dataset.max_corr[i, j] = max_corr_vec
        #else:
        #    for i in range(stacked_ctmaps.shape[0]):
        #        for j in range(stacked_ctmaps.shape[1]):
        #            for k in range(stacked_ctmaps.shape[2]):
        #                corr_vec = stacked_ctmaps[i, j, k, :]
        #                max_corr_vec_idx = np.argmax(corr_vec)
        #                max_corr_vec = corr_vec[max_corr_vec_idx]
        #                self.dataset.max_corr_idx[i, j, k] = max_corr_vec_idx
        #                self.dataset.max_corr[i, j, k] = max_corr_vec
        self.dataset.max_corr = np.max(stacked_ctmaps, axis=newaxis).todense()
        
        # TODO: This uses too much memory
        self.dataset.max_corr_idx = np.argmax(stacked_ctmaps.todense(), axis=newaxis)
        
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
