import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
import multiprocessing
import os
sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})
from functools import reduce
from sklearn.neighbors import KernelDensity
from sklearn import preprocessing
import scipy
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from multiprocessing import Pool
from contextlib import closing
from tempfile import mkdtemp, TemporaryDirectory
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
import community
import networkx as nx
from sklearn.cluster import DBSCAN
import sparse
from skimage import filters
from skimage.morphology import disk
from skimage import measure
from matplotlib.colors import ListedColormap
import pickle
import subprocess
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from PIL import Image
from scipy.ndimage import zoom
import pyarrow
import time
from packaging import version

from .utils import corr, calc_ctmap, calc_corrmap, flood_fill, calc_kde

def _fast_gaussian_kde(args):
    # TODO: 1) support sampling distance
    #       2) support other kernels
    (bandwidth, save_dir, gene_name, shape, locations, sampling_distance) = args
    
    print('Processing gene %s...'%gene_name)

    maxdist = int(bandwidth * 4)
    span = np.linspace(-maxdist,maxdist,maxdist*2+1)
    X, Y, Z = np.meshgrid(span,span,span)
    
    def create_kernel(x, y, z):
        X_=(-x+X)/bandwidth
        Y_=(-y+Y)/bandwidth
        Z_=(-z+Z)/bandwidth
        return np.exp(-0.5*(X_**2+Y_**2+Z_**2))
    
    pd = np.zeros(shape)
    for loc in locations:
        int_loc = [int(i) for i in loc]
        rem_loc = [i%1 for i in loc]

        kernel = create_kernel(*rem_loc)

        pos_start = [i - maxdist for i in int_loc]
        pos_end = [i + maxdist + 1 for i in int_loc]

        kernel_pos_start = [abs(i) if i < 0 else 0 for i in pos_start]
        kernel_pos_end = [maxdist*2+1 - (i-j) if i > j else maxdist*2+1 for i, j in zip(pos_end, shape)]

        pos_start = [0 if i < 0 else i for i in pos_start]
        pos_end = [j if i >= j else i for i, j in zip(pos_end, shape)]

        slices = tuple([slice(i, j) for i, j in zip(pos_start, pos_end)])
        kernel_slices = tuple([slice(i, j) for i, j in zip(kernel_pos_start, kernel_pos_end)])
        pd[slices] += kernel.swapaxes(0, 1)[kernel_slices]

    pd /= pd.sum()
    pd *= len(locations)
    
    return pd

def run_sctransform(data, clip_range=None, verbose=True, debug_path=None, plot_model_pars=False, **kwargs):
    """
    Run 'sctransform' R package and returns the normalized matrix and the model parameters.
    Package 'feather' is used for the data exchange between R and Python.
    :param data: N x D ndarray to normlize (N is number of samples, D is number of dimensions).
    :type data: numpy.ndarray
    :param kwargs: Any keyword arguments passed to R function `vst`.
    :returns: A 2-tuple, which contains two pandas.dataframe: 
        (1) normalized N x D matrix.
        (2) determined model parameters.
    """
    def _log(m):
        if verbose:
            print(m)
            
    vst_options = ['%s = "%s"'%(k, v) if type(v) is str else '%s = %s'%(k, v) for k, v in kwargs.items()]
    if len(vst_options) == 0:
        vst_opt_str = ''
    else:
        vst_opt_str = ', ' + ', '.join(vst_options)
    with TemporaryDirectory() as tmpdirname:
        if debug_path:
            tmpdirname = debug_path
        ifn, ofn, pfn, rfn = [os.path.join(tmpdirname, e) for e in ["in.feather", "out.feather", "fit_params.feather", "script.R"]]
        _log("Writing temporary files...")
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data, columns=[str(e) for e in range(data.shape[1])])
        if version.parse(pyarrow.__version__) >= version.parse("1.0.0"):
            df.to_feather(ifn, version=1)
        else:
            df.to_feather(ifn)
        rcmd = 'library(feather); library(sctransform); mat <- t(as.matrix(read_feather("{0}"))); colnames(mat) <- 1:ncol(mat); res <- vst(mat{1}, return_gene_attr=TRUE, return_cell_attr=TRUE); write_feather(as.data.frame(t(res$y)), "{2}"); write_feather(as.data.frame(res$model_pars_fit), "{3}");'.format(ifn, vst_opt_str, ofn, pfn)
        if plot_model_pars:
            plot_path = os.path.join(tmpdirname, 'model_pars.png')
            rcmd += 'png(file="%s", width=3600, height=1200, res=300); plot_model_pars(res, show_var=TRUE); dev.off();'%plot_path
        rcmd = rcmd.replace('\\', '\\\\')
        with open(rfn, "w") as f:
            f.write(rcmd)
        _log("Running scTransform via Rscript...")
        proc = subprocess.Popen(["Rscript", rfn], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while not proc.poll():
            c = proc.stdout.read(1)
            if not c:
                break
            if verbose:
                try:
                    sys.stdout.write(c.decode("utf-8"))
                except:
                    pass
            time.sleep(0.0001)
        _log("Reading output files...")
        o, p = pd.read_feather(ofn), pd.read_feather(pfn)
        if plot_model_pars:
            try:
                from matplotlib.image import imread
                import matplotlib.pyplot as plt
                img = imread(plot_path)
                dpi = 80
                fig = plt.figure(figsize=(img.shape[1]/dpi, img.shape[0]/dpi), dpi=dpi)
                plt.imshow(img, interpolation='nearest')
                plt.gca().set_axis_off()
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.show()
            except:
                print("Warning: plotting failed, perhaps matplotlib is not available?")
        _log("Clipping residuals...")
        if clip_range is None:
            r = np.sqrt(data.shape[0]/30.0)
            clip_range = (-r, r)
        o.clip(*clip_range)
        return o, p


class SSAMDataset(object):
    """
    A class to store intial values and results of SSAM analysis.

    :param genes: The genes that will be used for the analysis.
    :type genes: list(str)    
    :param locations: Location of the mRNAs in um, given as a list of
        N x D ndarrays (N is number of mRNAs, D is number of dimensions).
    :type locations: list(numpy.ndarray)
    :param width: Width of the image in um.
    :type width: float
    :param height: Height of the image in um.
    :type height: float
    :param depth: Depth of the image in um. Depth == 1 means 2D image.
    :type depth: float
    """
        
    def __init__(self, genes, locations, width, height, depth=1):
        if depth < 1 or width < 1 or height < 1:
            raise ValueError("Invalid image dimension")
        self.shape = (width, height, depth)
        self.ndim = 2 if depth == 1 else 3
        self.genes = list(genes)
        self.locations = []
        for l in list(locations):
            if l.shape[-1] == 3:
                self.locations.append(l)
            elif l.shape[-1] == 2:
                self.locations.append(np.concatenate((l, np.zeros([l.shape[0], 1])), axis=1))
            else:
                raise ValueError("Invalid mRNA locations")
        self.__vf = None
        self.__vf_norm = None
        self.normalized_vectors = None
        self.expanded_vectors = None
        self.cluster_labels = None
        #self.corr_map = None
        self.tsne = None
        self.umap = None
        self.normalized_vf = None
        self.excluded_clusters = None
        self.celltype_binned_counts = None

    @property
    def vf(self):
        """
        Vector field as a numpy.ndarray.
        """
        return self.__vf
    
    @vf.setter
    def vf(self, vf):
        self.__vf = vf
        self.__vf_norm = None
        
    @property
    def vf_norm(self):
        """
        `L1-norm <http://mathworld.wolfram.com/L1-Norm.html>`_ of the vector field as a numpy.ndarray.
        """

        if self.vf is None:
            return None
        if self.__vf_norm is None:
            self.__vf_norm = np.sum(self.vf, axis=len(self.vf.shape) - 1)
        return self.__vf_norm
    
    def plot_l1norm(self, cmap="viridis", rotate=0, z=None):
        """
        Plot the `L1-norm <http://mathworld.wolfram.com/L1-Norm.html>`_ of the vector field.

        :param cmap: Colormap used for the plot.
        :type cmap: str or matplotlib.colors.Colormap
        :param rotate: Rotate the plot. Possible values are 0, 1, 2, and 3.
        :type rotate: int
        :param z: Z index to slice 3D vector field.
            If not given, the slice at the middle will be plotted.
        :type z: int
        """
        if z is None:
            z = int(self.vf_norm.shape[2] / 2)
        if rotate < 0 or rotate > 3:
            raise ValueError("rotate can only be 0, 1, 2, 3")
        im = np.array(self.vf_norm, copy=True)
        if rotate == 1 or rotate == 3:
            im = im.swapaxes(0, 1)
        plt.imshow(im[..., z], cmap=cmap)
        if rotate == 1:
            plt.gca().invert_xaxis()
        elif rotate == 2:
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()
        elif rotate == 3:
            plt.gca().invert_yaxis()

    def plot_localmax(self, c=None, cmap=None, s=1, rotate=0):
        """
        Scatter plot the local maxima.

        :param c: Color of the scatter dots. Overrides `cmap` parameter.
        :type c: str or list(str), or list(float) or list(list(float))
        :param cmap: Colormap of the scatter dots.
        :type cmap: str or matplotlib.colors.Colormap
        :param s: Size of the scatter dots.
        :param rotate: Rotate the plot. Possible values are 0, 1, 2, and 3.
        :type rotate: int
        """
        if rotate < 0 or rotate > 3:
            raise ValueError("rotate can only be 0, 1, 2, 3")
        if rotate == 0 or rotate == 2:
            dim0, dim1 = 1, 0
        elif rotate == 1 or rotate == 3:
            dim0, dim1 = 0, 1
        plt.scatter(self.local_maxs[dim0], self.local_maxs[dim1], s=s, c=c, cmap=cmap)
        plt.xlim([0, self.vf_norm.shape[dim0]])
        plt.ylim([self.vf_norm.shape[dim1], 0])
        if rotate == 1:
            plt.gca().invert_xaxis()
        elif rotate == 2:
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()
        elif rotate == 3:
            plt.gca().invert_yaxis()    
        
    def __run_pca(self, exclude_bad_clusters, pca_dims, random_state):
        if exclude_bad_clusters:
            good_vecs = self.normalized_vectors[self.filtered_cluster_labels != -1, :]
        else:
            good_vecs = self.normalized_vectors
        return PCA(n_components=pca_dims, random_state=random_state).fit_transform(good_vecs)
        
    def plot_tsne(self, run_tsne=False, pca_dims=10, n_iter=5000, perplexity=70, early_exaggeration=10,
                  metric="correlation", exclude_bad_clusters=True, s=None, random_state=0, colors=[], excluded_color="#00000033", cmap="jet", tsne_kwargs={}):
        """
        Scatter plot the tSNE embedding.

        :param run_tsne: If false, this method tries to load precomputed tSNE result before running tSNE.
        :type run_tsne: bool
        :param pca_dims: Number of PCA dimensions used for the tSNE embedding.
        :type pca_dims: int
        :param n_iter: Maximum number of iterations for the tSNE.
        :type n_iter: int
        :param perplexity: The perplexity value of the tSNE (please refer to the section `How should I set the perplexity in t-SNE?` in this `link <https://lvdmaaten.github.io/tsne/>`_).
        :type perplexity: float
        :param early_exaggeration: Early exaggeration parameter for tSNE. Controls the tightness of the resulting tSNE plot.
        :type early_exaggeration: float
        :param metric: Metric for calculation of distance between vectors in gene expression space.
        :type metric: str
        :param exclude_bad_clusters: If true, the vectors that are excluded by the clustering algorithm will not be considered for tSNE computation.
        :type exclude_bad_clusters: bool
        :param s: Size of the scatter dots.
        :type s: float
        :param random_state: Random seed or scikit-learn's random state object to replicate the same result
        :type random_state: int or random state object
        :param colors: Color of each clusters.
        :type colors: list(str), list(list(float))
        :param excluded_color: Color of the vectors excluded by the clustering algorithm.
        :type excluded_color: str of list(float)
        :param cmap: Colormap for the clusters.
        :type cmap: str or matplotlib.colors.Colormap
        :param tsne_kwargs: Other keyward parameters for tSNE.
        :type tsne_kwargs: dict
        """
        if self.filtered_cluster_labels is None:
            exclude_bad_clusters = False
        if run_tsne or self.tsne is None:
            pcs = self.__run_pca(exclude_bad_clusters, pca_dims, random_state)
            self.tsne = TSNE(n_iter=n_iter, perplexity=perplexity, early_exaggeration=early_exaggeration, metric=metric, random_state=random_state, **tsne_kwargs).fit_transform(pcs[:, :pca_dims])
        if self.filtered_cluster_labels is not None:
            cols = self.filtered_cluster_labels[self.filtered_cluster_labels != -1]
        else:
            cols = None
        if len(colors) > 0:
            cmap = ListedColormap(colors)
        if not exclude_bad_clusters and self.filtered_cluster_labels is not None:
            plt.scatter(self.tsne[:, 0][self.filtered_cluster_labels == -1], self.tsne[:, 1][self.filtered_cluster_labels == -1], s=s, c=excluded_color)
            plt.scatter(self.tsne[:, 0][self.filtered_cluster_labels != -1], self.tsne[:, 1][self.filtered_cluster_labels != -1], s=s, c=cols, cmap=cmap)
        else:
            plt.scatter(self.tsne[:, 0], self.tsne[:, 1], s=s, c=cols, cmap=cmap)
        return

    def plot_umap(self, run_umap=False, pca_dims=10, metric="correlation", exclude_bad_clusters=True, s=None, random_state=0, colors=[], excluded_color="#00000033", cmap="jet", umap_kwargs={}):
        """
        Scatter plot the UMAP embedding.

        :param run_umap: If false, this method tries to load precomputed UMAP result before running UMAP.
        :type run_tsne: bool
        :param pca_dims: Number of PCA dimensions used for the UMAP embedding.
        :type pca_dims: int
        :param metric: Metric for calculation of distance between vectors in gene expression space.
        :type metric: str
        :param exclude_bad_clusters: If true, the vectors that are excluded by the clustering algorithm will not be considered for tSNE computation.
        :type exclude_bad_clusters: bool
        :param s: Size of the scatter dots.
        :type s: float
        :param random_state: Random seed or scikit-learn's random state object to replicate the same result
        :type random_state: int or random state object
        :param colors: Color of each clusters.
        :type colors: list(str), list(list(float))
        :param excluded_color: Color of the vectors excluded by the clustering algorithm.
        :type excluded_color: str of list(float)
        :param cmap: Colormap for the clusters.
        :type cmap: str or matplotlib.colors.Colormap
        :param umap_kwargs: Other keyward parameters for UMAP.
        :type umap_kwargs: dict
        """
        if self.filtered_cluster_labels is None:
            exclude_bad_clusters = False
        if run_umap or self.umap is None:
            pcs = self.__run_pca(exclude_bad_clusters, pca_dims, random_state)
            self.umap = UMAP(metric=metric, random_state=random_state, **umap_kwargs).fit_transform(pcs[:, :pca_dims])
        if self.filtered_cluster_labels is not None:
            cols = self.filtered_cluster_labels[self.filtered_cluster_labels != -1]
        else:
            cols = None
        if len(colors) > 0:
            cmap = ListedColormap(colors)
        if not exclude_bad_clusters and self.filtered_cluster_labels is not None:
            plt.scatter(self.umap[:, 0][self.filtered_cluster_labels == -1], self.umap[:, 1][self.filtered_cluster_labels == -1], s=s, c=excluded_color)
            plt.scatter(self.umap[:, 0][self.filtered_cluster_labels != -1], self.umap[:, 1][self.filtered_cluster_labels != -1], s=s, c=cols, cmap=cmap)
        else:
            plt.scatter(self.umap[:, 0], self.umap[:, 1], s=s, c=cols, cmap=cmap)
        return
    
    def plot_expanded_mask(self, cmap='Greys'): # TODO
        """
        Plot the expanded area of the vectors (Not fully implemented yet).

        :param cmap: Colormap for the mask.
        """
        plt.imshow(self.expanded_mask, vmin=0, vmax=1, cmap=cmap)
        return
    
    def plot_correlation_map(self, cmap='hot'): # TODO
        """
        Plot the correlations near the vectors in the vector field (Not fully implemented yet).

        :param cmap: Colormap for the image.
        """
        plt.imshow(self.corr_map, vmin=0.995, vmax=1.0, cmap=cmap)
        plt.colorbar()
        return
    
    def plot_celltypes_map(self, background="black", centroid_indices=[], colors=None, cmap='jet', rotate=0, min_r=0.6, set_alpha=False, z=None):
        """
        Plot the merged cell-type map.

        :param background: Set background color of the cell-type map.
        :type background: str or list(float)
        :param centroid_indices: The centroids which will be in the cell type map. If not given, the cell-type map is drawn with all centroids.
        :type centroid_indices: list(int)
        :param colors: Color of the clusters. Overrides `cmap` parameter.
        :type colors: list(str), list(list(float))
        :param cmap: Colormap for the clusters.
        :type cmap: str or matplotlib.colors.Colormap
        :param rotate: Rotate the plot. Possible values are 0, 1, 2, and 3.
        :type rotate: int
        :param min_r: Minimum correlation threshold for the cell-type map.
            This value is only for the plotting, does not affect to the cell-type maps generated by `filter_celltypemaps`.
        :type min_r: float
        :param set_alpha: Set alpha of each pixel based on the correlation.
            Not properly implemented yet, doesn't work properly with the background other than black.
        :type set_alpha: bool
        :param z: Z index to slice 3D cell-type map.
            If not given, the slice at the middle will be used.
        :type z: int
        """
        if z is None:
            z = int(self.shape[2] / 2)
        num_ctmaps = np.max(self.filtered_celltype_maps) + 1
        
        if len(centroid_indices) == 0:
            centroid_indices = list(range(num_ctmaps))
            
        if colors is None:
            cmap_internal = plt.get_cmap(cmap)
            colors = cmap_internal([float(i) / (num_ctmaps - 1) for i in range(num_ctmaps)])
            
        all_colors = [background if not j in centroid_indices else colors[i] for i, j in enumerate(range(num_ctmaps))]
        cmap_internal = ListedColormap(all_colors)

        celltype_maps_internal = np.array(self.filtered_celltype_maps[..., z], copy=True)
        empty_mask = celltype_maps_internal == -1
        celltype_maps_internal[empty_mask] = 0
        sctmap = cmap_internal(celltype_maps_internal)
        sctmap[empty_mask] = (0, 0, 0, 0)

        if set_alpha:
            alpha = np.array(self.max_correlations[..., z], copy=True)
            alpha[alpha < 0] = 0 # drop negative correlations
            alpha = min_r + alpha / (np.max(alpha) / (1.0 - min_r))
            sctmap[..., 3] = alpha

        if rotate == 1 or rotate == 3:
            sctmap = sctmap.swapaxes(0, 1)

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

    def plot_domains(self, background='white', colors=None, cmap='jet', rotate=0, domain_background=False, background_alpha=0.3, z=None):
        """
        Plot tissue domains.

        :param background: Background color of the plot.
        :type background: str or list(float)
        :param colors: Color of the domains. Overrides `cmap` parameter.
        :type colors: list(str), list(list(float))
        :param cmap: Colormap for the domains.
        :type cmap: str or matplotlib.colors.Colormap
        :param rotate: Rotate the plot. Possible values are 0, 1, 2, and 3.
        :type rotate: int
        :param domain_background: Show the area of the inferred domains behind the domain map.
        :type domain_background: bool
        :param background_alpha: The alpha value of the area of the inferred domains.
        :type background_alpha: float
        :param z: Z index to slice 3D domain map.
            If not given, the slice at the middle will be used.
        :type z: int
        """
        if z is None:
            z = int(self.shape[2] / 2)
        
        inferred_domains = self.inferred_domains[..., z]
        inferred_domains_cells = self.inferred_domains_cells[..., z]
        
        if rotate == 1 or rotate == 3:
            inferred_domains = inferred_domains.swapaxes(0, 1)
            inferred_domains_cells = inferred_domains_cells.swapaxes(0, 1)
            
        if colors is None:
            cmap_internal = plt.get_cmap(cmap)
            colors_domains = cmap_internal(np.linspace(0, 1, np.max(inferred_domains) + 1))
            colors_cells = cmap_internal(np.linspace(0, 1, np.max(inferred_domains_cells) + 1))
            
        colors_domains[:, 3] = background_alpha
        if -1 in inferred_domains:
            colors_domains = [[0, 0, 0, 0]] + list(colors_domains)
        if -1 in inferred_domains_cells:
            colors_cells = [[0, 0, 0, 0]] + list(colors_cells)
            
        plt.gca().set_facecolor(background)
        if domain_background:
            plt.imshow(inferred_domains, cmap=ListedColormap(colors_domains))
        plt.imshow(inferred_domains_cells, cmap=ListedColormap(colors_cells))
        
        if rotate == 1:
            plt.gca().invert_xaxis()
        elif rotate == 2:
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()
        elif rotate == 3:
            plt.gca().invert_yaxis()
            
        return
    
    def plot_diagnostic_plot(self, centroid_index, cluster_name=None, cluster_color=None, cmap=None, rotate=0, z=None, use_embedding="tsne", known_signatures=[], correlation_methods=[]):
        """
        Plot the diagnostic plot. This method requires `plot_tsne` or `plot_umap` was run at least once before.

        :param centroid_index: Index of the centroid for the diagnostic plot.
        :type centroid_index: int
        :param cluster_name: The name of the cluster.
        :type cluster_name: str
        :param cluster_color: The color of the cluster. Overrides `cmap` parameter.
        :type cluster_color: str or list(float)
        :param cmap: The colormap for the clusters. The cluster color is determined using the `centroid_index` th color of the given colormap.
        :type cmap: str or matplotlib.colors.Colormap
        :param rotate: Rotate the plot. Possible values are 0, 1, 2, and 3.
        :type rotate: int
        :param z: Z index to slice 3D vector norm and cell-type map plots.
            If not given, the slice at the middle will be used.
        :type z: int
        :param use_embedding: The type of the embedding for the last panel. Possible values are "tsne" or "umap".
        :type use_embedding: str
        :param known_signatures: The list of known signatures, which will be displayed in the 3rd panel. Each signature can be 3-tuple or 4-tuple,
            containing 1) the name of signature, 2) gene labels of the signature, 3) gene expression values of the signature, 4) optionally the color of the signature.
        :type known_signatures: list(tuple)
        :param correlation_methods: The correlation method used to determine max correlation of the centroid to the `known_signatures`. Each method should be 2-tuple,
            containing 1) the name of the correaltion, 2) the correaltion function (compatiable with the correlation methods available in `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`_)
        :type correlation_methods: list(tuple)
        """
        if z is None:
            z = int(self.vf_norm.shape[2] / 2)
        p, e = self.centroids[centroid_index], self.centroids_stdev[centroid_index]
        if cluster_name is None:
            cluster_name = "Cluster #%d"%centroid_index
        
        if cluster_color is None:
            if cmap is None:
                cmap = plt.get_cmap("jet")
            cluster_color = cmap(centroid_index / (len(self.centroids) - 1))

        if len(correlation_methods) == 0:
            correlation_methods = [("r", corr), ]
        total_signatures = len(correlation_methods) * len(known_signatures) + 1
                
        ax = plt.subplot(1, 4, 1)
        mask = self.filtered_cluster_labels == centroid_index
        plt.scatter(self.local_maxs[0][mask], self.local_maxs[1][mask], c=[cluster_color])
        self.plot_l1norm(rotate=rotate, cmap="Greys", z=z)

        ax = plt.subplot(1, 4, 2)
        ctmap = np.zeros([self.filtered_celltype_maps.shape[0], self.filtered_celltype_maps.shape[1], 4])
        ctmap[self.filtered_celltype_maps[..., z] == centroid_index] = to_rgba(cluster_color)
        ctmap[np.logical_and(self.filtered_celltype_maps[..., z] != centroid_index, self.filtered_celltype_maps[..., 0] > -1)] = [0.9, 0.9, 0.9, 1]
        if rotate == 1 or rotate == 3:
            ctmap = ctmap.swapaxes(0, 1)
        ax.imshow(ctmap)
        if rotate == 1:
            ax.invert_xaxis()
        elif rotate == 2:
            ax.invert_xaxis()
            ax.invert_yaxis()
        elif rotate == 3:
            ax.invert_yaxis()

        ax = plt.subplot(total_signatures, 4, 3)
        ax.bar(self.genes, p, yerr=e)
        ax.set_title(cluster_name)
        plt.xlim([-1, len(self.genes)])
        plt.xticks(rotation=90)

        subplot_idx = 0
        for signature in known_signatures:
            sig_title, sig_labels, sig_values = signature[:3]
            sig_colors_defined = False
            if len(signature) == 4:
                sig_colors = signature[3]
                sig_colors_defined = True
            for corr_label, corr_func in correlation_methods:
                corr_results = [corr_func(p, sig_value) for sig_value in sig_values]
                corr_results = [e[0] if hasattr(e, "__getitem__") else e for e in corr_results]
                max_corr_idx = np.argmax(corr_results)
                ax = plt.subplot(total_signatures, 4, 7+subplot_idx*4)
                lbl = sig_labels[max_corr_idx]
                if sig_colors_defined:
                    col = sig_colors[max_corr_idx]
                else:
                    col = cluster_color
                ax.bar(self.genes, sig_values[max_corr_idx], color=col)
                ax.set_title("%s in %s (max %s, %.3f)"%(lbl, sig_title, corr_label, corr_results[max_corr_idx]))
                plt.xlim([-1, len(self.genes)])
                plt.xticks(rotation=90)
                subplot_idx += 1

        if use_embedding == 'tsne':
            embedding = self.tsne
            fig_title = "t-SNE, %d vectors"%sum(self.filtered_cluster_labels == centroid_index)
        elif use_embedding == 'umap':
            embedding = self.umap
            fig_title = "UMAP, %d vectors"%sum(self.filtered_cluster_labels == centroid_index)
        good_vectors = self.filtered_cluster_labels[self.filtered_cluster_labels != -1]
        ax = plt.subplot(1, 4, 4)
        ax.scatter(embedding[:, 0][good_vectors != centroid_index], embedding[:, 1][good_vectors != centroid_index], c=[[0.8, 0.8, 0.8, 1],], s=80)
        ax.scatter(embedding[:, 0][good_vectors == centroid_index], embedding[:, 1][good_vectors == centroid_index], c=[cluster_color], s=80)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(fig_title)
        
    def plot_celltype_composition(self, domain_index, cell_type_colors=None, cell_type_cmap='jet', cell_type_orders=None, label_cutoff=0.03, pctdistance=1.15, **kwargs):
        """
        Plot composition of cell types in each domain.

        :param domain_index: Index of the domain.
        :type domain_index: int
        :param cell_type_colors: The colors of the cell types. Overrides `cell_type_cmap` parameter.
        :type cell_type_colors: str or list(float)
        :param cell_type_cmap: The colormap for the cell types.
        :type cell_type_cmap: str or matplotlib.colors.Colormap
        :param label_cutoff: The minimum cutoff of the labeling of the percentage. From 0 to 1.
        :type label_cutoff: float
        :param pctdistance: The distance from center of the pie to the labels.
        :type pctdistance: float
        :param kwargs: More kewward arguments for the matplotlib.pyplot.pie.
        """
        if cell_type_colors is None:
            cmap = plt.get_cmap(cell_type_cmap)
            cell_type_colors = cmap(np.arange(0, len(self.centroids)) / (len(self.centroids) - 1))
        
        if cell_type_orders is not None:
            ctcs = np.array(cell_type_colors)[cell_type_orders]
            p = self.inferred_domains_compositions[domain_index][cell_type_orders]
        else:
            ctcs = cell_type_colors
            p = self.inferred_domains_compositions[domain_index]
        plt.pie(p,
                colors=ctcs,
                autopct=lambda e: '%.1f %%'%e if e > 3 else '',
                pctdistance=pctdistance, **kwargs)

    def plot_spatial_relationships(self, cluster_labels, *args, **kwargs):
        """
        Plot spatial relationship between cell types, presented as a heatmap.

        :param cluster_labels: x- and y-axis label of the heatmap.
        :type cluster_labels: list(str)
        :param args: More arguments for the seaborn.heatmap.
        :param kwargs: More keyword arguments for the seaborn.heatmap.
        """
        sns.heatmap(self.spatial_relationships, *args, xticklabels=cluster_labels, yticklabels=cluster_labels, **kwargs)    

    def get_celltype_correlation(self, idx):
        """
        Get correlation values of a cell type map between the given cluster's centroid to the vector field.
        
        :param idx: Index of a cluster
        :type idx: int
        :return: Correlation values of a cell type map of the specified cluster's centroid
        :rtype: numpy.ndarray
        """
        rtn = np.zeros_like(self.max_correlations) - 1
        rtn[self.celltype_maps == idx] = self.max_correlations[self.celltype_maps == idx]
        return rtn
    
        
class SSAMAnalysis(object):
    """
    A class to run SSAM analysis.

    :param dataset: A SSAMDataset object.
    :type dataset: SSAMDataset
    :param ncores: Number of cores for parallel computation. If a negative value is given,
        ((# of all available cores on system) - abs(ncores)) cores will be used.
    :type ncores: int
    :param save_dir: Directory to store intermediate data (e.g. density / vector field).
        Any data which already exists will be loaded and reused.
    :type save_dir: str
    :param verbose: If True, then it prints out messages during the analysis.
    :type verbose: bool
    """
    def __init__(self, dataset, ncores=-1, save_dir="", verbose=False):
        
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
            
    def run_kde(self, kernel="gaussian", bandwidth=2.5, sampling_distance=1.0, use_mmap=False):
        """
        Run KDE to estimate density of mRNA.

        :param kernel: Kernel for density estimation.
        :type kernel: str
        :param bandwidth: Parameter to adjust width of kernel.
            Set it 2.5 to make FWTM of Gaussian kernel to be ~10um (assume that avg. cell diameter is ~10um).
        :type bandwidth: float
        :param sampling_distance: Grid spacing in um.
        :type sampling_distance: float
        :param use_mmap: Use MMAP to reduce memory usage during analysis.
            Turning on this option can reduce the amount of memory used by SSAM analysis, but also lower the analysis speed.
        :type use_mmap: bool
        """
        def save_pickle(fn, o):
            with open(fn, "wb") as f:
                return pickle.dump(o, f, protocol=4)
        def load_pickle(fn):
            with open(fn, "rb") as f:
                return pickle.load(f)
        
        steps = [int(np.ceil(e / sampling_distance)) for e in self.dataset.shape]
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
                        for z in range(steps[2]):
                            chunk[cnt, :] = [x, y, z]
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
                    if kernel != "gaussian":
                        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(self.dataset.locations[gidx])
                        if pool is None:
                            pool = multiprocessing.Pool(self.ncores)
                    else:
                        X, Y, Z = [self.dataset.locations[gidx][:, i] for i in range(3)]
                    for chunks in yield_chunks():
                        if kernel == "gaussian":
                            pdf_chunks = [calc_kde(bandwidth, X, Y, Z, chunk[:, 0], chunk[:, 1], chunk[:, 2], 0, self.ncores) for chunk in chunks]
                        else:
                            pdf_chunks = pool.map(kde.score_samples, [chunk * sampling_distance for chunk in chunks])
                        for pdf_chunk, pos_chunk in zip(pdf_chunks, chunks):
                            if kernel == "gaussian":
                                pdf[pos_chunk[:, 0], pos_chunk[:, 1], pos_chunk[:, 2]] = pdf_chunk
                            else:
                                pdf[pos_chunk[:, 0], pos_chunk[:, 1], pos_chunk[:, 2]] = np.exp(pdf_chunk)
                    pdf /= np.sum(pdf)
                    np.save(pdf_filename, pdf)
                vf[..., gidx] = pdf * len(self.dataset.locations[gidx])
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
   
    def run_fast_kde(self, kernel='gaussian', bandwidth=2.5, sampling_distance=1.0, re_run=False, use_mmap=False):
        """
        Run KDE faster than `run_kde` method. This method uses precomputed kernels to estimate density of mRNA.

        :param kernel: Kernel for density estimation. Currently only Gaussian kernel is supported.
        :type kernel: str
        :param bandwidth: Parameter to adjust width of kernel.
            Set it 2.5 to make FWTM of Gaussian kernel to be ~10um (assume that avg. cell diameter is ~10um).
        :type bandwidth: float
        :param sampling_distance: Grid spacing in um. Currently only 1 um is supported.
        :type sampling_distance: float
        :param re_run: Recomputes KDE, ignoring all existing precomputed densities in the data directory.
        :type re_run: bool
        :param use_mmap: Use MMAP to reduce memory usage during analysis. Currently not implemented, this option should be always disabled.
        :type use_mmap: bool
        """
        if kernel != 'gaussian':
            raise NotImplementedError('Only Gaussian kernel is supported.')
        if sampling_distance != 1.0:
            raise NotImplementedError('Sampling distance should be 1.')
        if use_mmap:
            raise NotImplementedError('MMAP is not supported yet.')
            
        def save_pickle(fn, o):
            with open(fn, "wb") as f:
                return pickle.dump(o, f, protocol=4)
        def load_pickle(fn):
            with open(fn, "rb") as f:
                return pickle.load(f)

        vf_filename = os.path.join(self.save_dir, 'vf_sd%s_bw%s.pkl'%(
            ('%f' % sampling_distance).rstrip('0').rstrip('.'),
            ('%f' % bandwidth).rstrip('0').rstrip('.')
        ))
        
        if os.path.exists(vf_filename) and not re_run:
            self.dataset.vf = load_pickle(vf_filename)
            return

        self.dataset.vf = np.zeros(self.dataset.shape+(len(self.dataset.genes),))
        idcs = np.argsort([len(i) for i in self.dataset.locations])[::-1]
        pdf_filenames = [os.path.join(self.save_dir, 'pdf_sd%s_bw%s_%s.npy'%(
            ('%f' % sampling_distance).rstrip('0').rstrip('.'),
            ('%f' % bandwidth).rstrip('0').rstrip('.'),
            self.dataset.genes[gidx])
        ) for gidx in idcs]

        if not re_run:
            idcs = np.where([not os.path.exists(fn) for fn in pdf_filenames])[0]
            for gidx in np.where([os.path.exists(fn) for fn in pdf_filenames])[0]:
                print("Loading gene %s..."%self.dataset.genes[gidx])
                self.dataset.vf[..., gidx] = np.load(pdf_filenames[gidx])
        
        if len(idcs) > 0:
            with closing(Pool(self.ncores, maxtasksperchild=1)) as p:
                res = p.imap(_fast_gaussian_kde,[(bandwidth,
                                                 self.save_dir,
                                                 self.dataset.genes[gidx],
                                                 self.dataset.shape,
                                                 self.dataset.locations[gidx],
                                                 sampling_distance) for gidx in idcs])
                for gidx, pd in zip(idcs, res): # imap returns result in the same order as the input array
                    self.dataset.vf[..., gidx] = pd
                    np.save(pdf_filenames[gidx], pd)
                p.close()
                p.join()
        save_pickle(vf_filename, self.dataset.vf)

    def calc_correlation_map(self, corr_size=3):
        """
        Calculate local correlation map of the vector field.

        :param corr_size: Size of square (or cube) that is used to compute the local correlation values.
            This value should be an odd number.
        :type corr_size: int
        """
        
        corr_map = calc_corrmap(self.dataset.vf, ncores=self.ncores, size=int(corr_size/2))
        self.dataset.corr_map = np.array(corr_map, copy=True)
        return
    
    def find_localmax(self, search_size=3, min_norm=0, min_expression=0, mask=None):
        """
        Find local maxima vectors in the norm of the vector field.

        :param search_size: Size of square (or cube in 3D) that is used to search for the local maxima.
            This value should be an odd number.
        :type search_size: int
        :param min_norm: Minimum value of norm at the local maxima.
        :type min_norm: float
        :param min_expression: Minimum value of gene expression in a unit pixel at the local maxima.
            mask: numpy.ndarray, optional
            If given, find vectors in the masked region, instead of the whole image.
        :type min_expression: float
        """

        max_mask = self.dataset.vf_norm == ndimage.maximum_filter(self.dataset.vf_norm, size=search_size)
        max_mask &= self.dataset.vf_norm > min_norm
        if min_expression > 0:
            exp_mask = np.zeros_like(max_mask)
            for i in range(len(self.dataset.genes)):
                exp_mask |= self.dataset.vf[..., i] > min_expression
            max_mask &= exp_mask
        if mask is not None:
            max_mask &= mask
        local_maxs = np.where(max_mask)
        self.__m__("Found %d local max vectors."%len(local_maxs[0]))
        self.dataset.local_maxs = local_maxs
        return

    def expand_localmax(self, r=0.99, min_pixels=7, max_pixels=1000):
        """
        Merge the vectors nearby the local max vectors.
        Only the vectors with the large Pearson correlation values are merged.

        :param r: Minimum Pearson's correlation coefficient to look for the nearby vectors.
        :type r: float
        :param min_pixels: Minimum number of pixels to merge.
        :type min_pixels: float
        :param max_pixels: Maximum number of pixels to merge.
        :type max_pixels: float
        """
        
        expanded_vecs = []
        self.__m__("Expanding local max vectors...")
        fill_dx = np.meshgrid(range(3), range(3), range(3))
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
    
    def normalize_vectors_sctransform(self, use_expanded_vectors=False, normalize_vf=True, vst_kwargs={}):
        """
        Normalize and regularize vectors using SCtransform

        :param use_expanded_vectors: If True, use averaged vectors nearby local maxima
            of the vector field.
        :type use_expanded_vectors: bool
        :param normalize_vf: If True, the vector field is also normalized
            using the same parameters used to normalize the local maxima.
        :type normalize_vf: bool
        :param vst_kwargs: Optional keywords arguments for sctransform's vst function.
        :type vst_kwargs: dict
        """
        if use_expanded_vectors:
            vec = np.array(self.dataset.expanded_vectors, copy=True)
        else:
            vec = np.array(self.dataset.vf[self.dataset.local_maxs], copy=True)
        
        norm_vec, fit_params = run_sctransform(vec, **vst_kwargs)
        self.dataset.normalized_vectors = np.array(norm_vec)
        
        if normalize_vf:
            vf_nonzero = self.dataset.vf[self.dataset.vf_norm > 0]
            nvec = vf_nonzero.shape[0]
            fit_params = np.array(fit_params).T
            regressor_data = np.ones([nvec, 2])
            regressor_data[:, 1] = np.log10(np.sum(vf_nonzero, axis=1))
            
            mu = np.exp(np.dot(regressor_data, fit_params[1:, :]))
            with np.errstate(divide='ignore', invalid='ignore'):
                res = (vf_nonzero - mu) / np.sqrt(mu + mu**2 / fit_params[0, :])
            self.dataset.normalized_vf = np.zeros_like(self.dataset.vf)
            self.dataset.normalized_vf[self.dataset.vf_norm > 0] = np.nan_to_num(res)
        return
    
    def normalize_vectors(self, use_expanded_vectors=False, normalize_gene=False, normalize_vector=False, normalize_median=False, size_after_normalization=1e4, log_transform=False, scale=False):
        """
        Normalize and regularize vectors

        :param use_expanded_vectors: If True, use averaged vectors nearby local maxima of the vector field.
        :type use_expanded_vectors: bool
        :param normalize_gene: If True, normalize vectors by sum of each gene expression across all vectors.
        :type normalize_gene: bool
        :param normalize_vector: If True, normalize vectors by sum of all gene expression of each vector.
        :type normalize_vector: bool
        :param log_transform: If True, vectors are log transformed.
        :type log_transform: bool
        :param scale: If True, vectors are z-scaled (mean centered and scaled by stdev).
        :type scale: bool
        """
        if use_expanded_vectors:
            vec = np.array(self.dataset.expanded_vectors, copy=True)
        else:
            vec = np.array(self.dataset.vf[self.dataset.local_maxs], copy=True)
        if normalize_gene:
            vec = preprocessing.normalize(vec, norm=norm, axis=0) * size_after_normalization  # Normalize per gene
        if normalize_vector:
            vec = preprocessing.normalize(vec, norm="l1", axis=1) * size_after_normalization # Normalize per vector
        if normalize_median:
            def n(v):
                s, m = np.sum(v, axis=1), np.median(v, axis=1)
                s[m > 0] = s[m > 0] / m[m > 0]
                s[m == 0] = 0
                v[s > 0] = v[s > 0] / s[s > 0][:, np.newaxis]
                v[v == 0] = 0
                return v
            vec = n(vec)
        if log_transform:
            vec = np.log2(vec + 1)
        if scale:
            vec = preprocessing.scale(vec)
        self.dataset.normalized_vectors = vec
        return
    
    def __correct_cluster_labels(self, cluster_labels, centroid_correction_threshold):
        new_labels = np.array(cluster_labels, copy=True)
        if centroid_correction_threshold < 1.0:
            for cidx in np.unique(cluster_labels):
                if cidx == -1:
                    continue
                prev_midx = -1
                while True:
                    vecs = self.dataset.normalized_vectors[new_labels == cidx]
                    vindices = np.where(new_labels == cidx)[0]
                    midx = vindices[np.argmin(np.sum(cdist(vecs, vecs), axis=0))]
                    if midx == prev_midx:
                        break
                    prev_midx = midx
                    m = self.dataset.normalized_vectors[midx]
                    for vidx, v in zip(vindices, vecs):
                        if corr(v, m) < centroid_correction_threshold:
                            new_labels[vidx] = -1
        return new_labels

    def __calc_centroid(self, cluster_labels):
        centroids = []
        centroids_stdev = []
        #medoids = []
        for lbl in sorted(list(set(cluster_labels))):
            if lbl == -1:
                continue
            cl_vecs = self.dataset.normalized_vectors[cluster_labels == lbl, :]
            #cl_dists = scipy.spatial.distance.cdist(cl_vecs, cl_vecs, metric)
            #medoid = cl_vecs[np.argmin(np.sum(cl_dists, axis=0))]
            centroid = np.mean(cl_vecs, axis=0)
            centroid_stdev = np.std(cl_vecs, axis=0)
            #medoids.append(medoid)
            centroids.append(centroid)
            centroids_stdev.append(centroid_stdev)
        return centroids, centroids_stdev#, medoids

    def cluster_vectors(self, pca_dims=10, min_cluster_size=0, resolution=0.6, prune=1.0/15.0, snn_neighbors=30, max_correlation=1.0,
                        metric="correlation", subclustering=False, dbscan_eps=0.4, centroid_correction_threshold=0.8, random_state=0):
        """
        Cluster the given vectors using the specified clustering method.

        :param pca_dims: Number of principal componants used for clustering.
        :type pca_dims: int
        :param min_cluster_size: Set minimum cluster size.
        :type min_cluster_size: int
        :param resolution: Resolution for Louvain community detection.
        :type resolution: float
        :param prune: Threshold for Jaccard index (weight of SNN network). If it is smaller than prune, it is set to zero.
        :type prune: float
        :param snn_neighbors: Number of neighbors for SNN network.
        :type snn_neighbors: int
        :param max_correlation: Clusters with higher correlation to this value will be merged.
        :type max_correlation: bool
        :param metric: Metric for calculation of distance between vectors in gene expression space.
        :type metric: str
        :param subclustering: If True, each cluster will be clustered once again with DBSCAN algorithm to find more subclusters.
        :type subclustering: bool
        :param centroid_correction_threshold: Centroid will be recalculated with the vectors
            which have the correlation to the cluster medoid equal or higher than this value.
        :type centroid_correction_threshold: float
        :param random_state: Random seed or scikit-learn's random state object to replicate the same result
        :type random_state: int or random state object
        """
        
        vecs_normalized = self.dataset.normalized_vectors
        vecs_normalized_dimreduced = PCA(n_components=pca_dims, random_state=random_state).fit_transform(vecs_normalized)

        def cluster_vecs(vecs):
            k = min(snn_neighbors, vecs.shape[0])
            knn_graph = kneighbors_graph(vecs, k, mode='connectivity', include_self=True, metric=metric).todense()
            intersections = np.dot(knn_graph, knn_graph.T)
            snn_graph = intersections / (k + (k - intersections)) # borrowed from Seurat
            snn_graph[snn_graph < prune] = 0
            G = nx.from_numpy_matrix(snn_graph)
            partition = community.best_partition(G, resolution=resolution, random_state=random_state)
            lbls = np.array(list(partition.values()))
            return lbls

        def remove_small_clusters(lbls, lbls2=None):
            small_clusters = []
            cluster_indices = []
            lbls = np.array(lbls)
            for lbl in np.unique(lbls):
                if lbl == -1:
                    continue
                cnt = np.sum(lbls == lbl)
                if cnt < min_cluster_size:
                    small_clusters.append(lbl)
                else:
                    cluster_indices.append(lbl)
            for lbl in small_clusters:
                lbls[lbls == lbl] = -1
            tmp = np.array(lbls, copy=True)
            for i, idx in enumerate(cluster_indices):
                lbls[tmp == idx] = i
            if lbls2 is not None:
                for lbl in small_clusters:
                    lbls2[lbls2 == lbl] = -1
                tmp = np.array(lbls2, copy=True)
                for i, idx in enumerate(cluster_indices):
                    lbls2[tmp == idx] = i
                return lbls, lbls2
            else:
                return lbls
        
        if subclustering:
            super_lbls = cluster_vecs(vecs_normalized_dimreduced)
            dbscan = DBSCAN(eps=dbscan_eps, min_samples=min_cluster_size, metric=metric)
            all_lbls = np.zeros_like(super_lbls)
            global_lbl_idx = 0
            for super_lbl in set(list(super_lbls)):
                super_lbl_idx = np.where(super_lbls == super_lbl)[0]
                if super_lbl == -1:
                    all_lbls[super_lbl_idx] = -1
                    continue
                sub_lbls = dbscan.fit(vecs_normalized_dimreduced[super_lbl_idx]).labels_
                for sub_lbl in set(list(sub_lbls)):
                    if sub_lbl == -1:
                        all_lbls[tuple([super_lbl_idx[sub_lbls == sub_lbl]])] = -1
                        continue
                    all_lbls[tuple([super_lbl_idx[sub_lbls == sub_lbl]])] = global_lbl_idx
                    global_lbl_idx += 1
        else:
            all_lbls = cluster_vecs(vecs_normalized_dimreduced)            
                
        new_labels = self.__correct_cluster_labels(all_lbls, centroid_correction_threshold)
        new_labels, all_lbls = remove_small_clusters(new_labels, all_lbls)
        centroids, centroids_stdev = self.__calc_centroid(new_labels)

        merge_candidates = []
        if max_correlation < 1.0:
            Z = scipy.cluster.hierarchy.linkage(centroids, metric='correlation')
            clbls = scipy.cluster.hierarchy.fcluster(Z, 1 - max_correlation, 'distance')
            for i in set(clbls):
                leaf_indices = np.where(clbls == i)[0]
                if len(leaf_indices) > 1:
                    merge_candidates.append(leaf_indices)
            removed_indices = []
            for cand in merge_candidates:
                for i in cand[1:]:
                    all_lbls[all_lbls == i] = cand[0]
                    removed_indices.append(i)
            for i in sorted(removed_indices, reverse=True):
                all_lbls[all_lbls > i] -= 1

            new_labels = self.__correct_cluster_labels(all_lbls, centroid_correction_threshold)
            new_labels, all_lbls = remove_small_clusters(new_labels, all_lbls)
            centroids, centroids_stdev = self.__calc_centroid(new_labels)
                
        self.dataset.cluster_labels = all_lbls
        self.dataset.filtered_cluster_labels = new_labels
        self.dataset.centroids = np.array(centroids)
        self.dataset.centroids_stdev = np.array(centroids_stdev)
        #self.dataset.medoids = np.array(medoids)
        
        self.__m__("Found %d clusters"%len(centroids))
        return

    def rescue_cluster(self, gene_names, expression_thresholds=[]):
        assert len(gene_names) > 0
        assert len(expression_thresholds) == 0 or len(gene_names) == len(expression_thresholds)

        expression_thresholds = list(expression_thresholds)
        lm_vectors = self.dataset.vf[self.dataset.local_maxs[0], self.dataset.local_maxs[1], self.dataset.local_maxs[2], :]
        lm_mask = np.ones(len(lm_vectors), dtype=bool)
        for i in range(len(gene_names)):
            rg_idx = self.dataset.genes.index(gene_names[i])
            if len(expression_thresholds) == 0:
                expression_threshold = filters.threshold_otsu(self.dataset.vf[..., rg_idx])
            else:
                expression_threshold = float(expression_thresholds[i])
            lm_mask = np.logical_and(lm_mask, lm_vectors[:, rg_idx] > expression_threshold)

        rg_vectors = lm_vectors[lm_mask]
        rg_centroid = np.mean(rg_vectors, axis=0)
        rg_centroid_stdev = np.std(rg_vectors, axis=0)

        self.dataset.cluster_labels[lm_mask] = len(self.dataset.centroids)
        self.dataset.filtered_cluster_labels[lm_mask] = len(self.dataset.centroids)
        self.dataset.centroids = np.append(self.dataset.centroids, [rg_centroid], axis=0)
        self.dataset.centroids_stdev = np.append(self.dataset.centroids_stdev, [rg_centroid_stdev], axis=0)
    
    def exclude_and_merge_clusters(self, exclude=[], merge=[], centroid_correction_threshold=0.8):
        """
        Exclude bad clusters (including the vectors in the clusters), and merge similar clusters for the downstream analysis.

        :param exclude: List of cluster indices to be excluded.
        :type exclude: list(int)
        :param merge: List of list of cluster indices to be merged.
        :type merge: list(list(int))
        :param centroid_correction_threshold: Centroid will be recalculated with the vectors
            which have the correlation to the cluster medoid equal or higher than this value.
        :type centroid_correction_threshold: float
        """
        exclude = list(exclude)
        merge = np.array(merge)
        for centroids in merge:
            centroids = np.unique(centroids)
            for centroid in centroids[1:][::-1]:
                self.dataset.cluster_labels[self.dataset.cluster_labels == centroid] = centroids[0]
                exclude.append(centroid)
        exclude = sorted(exclude)
        
        mask = np.ones(len(self.dataset.centroids), np.bool)
        mask[exclude] = False

        #self.dataset.centroids = self.dataset.centroids[mask]
        #self.dataset.centroids_stdev = self.dataset.centroids_stdev[mask]
        #self.dataset.medoids = self.dataset.medoids[mask]

        mask = np.ones(len(self.dataset.cluster_labels), np.bool)
        for centroid in exclude:
            # There will be no vectors for already merged centroids - so there is no problem
            mask[np.array(self.dataset.cluster_labels) == centroid] = False
        self.dataset.cluster_labels = self.dataset.cluster_labels[mask]
        self.dataset.local_maxs = tuple([lm[mask] for lm in self.dataset.local_maxs])
        
        for centroid in exclude[::-1]:
            self.dataset.cluster_labels[self.dataset.cluster_labels > centroid] -= 1
        self.dataset.normalized_vectors = self.dataset.normalized_vectors[mask, :]
        
        new_labels = self.__correct_cluster_labels(self.dataset.cluster_labels, centroid_correction_threshold)
        centroids, centroids_stdev = self.__calc_centroid(new_labels)
        
        self.dataset.centroids = centroids
        self.dataset.centroids_stdev = centroids_stdev
        self.dataset.filtered_cluster_labels = new_labels
        
        return
    
    def map_celltypes(self, centroids=None):
        """
        Create correlation maps between the centroids and the vector field.
        Each correlation map corresponds each cell type map.

        :param centroids: If given, map celltypes with the given cluster centroids.
        :type centroids: list(np.array(int))
        """
        
        if self.dataset.normalized_vf is None:
            normalized_vf = self.dataset.vf
        else:
            normalized_vf = self.dataset.normalized_vf

        if centroids is None:
            centroids = self.dataset.centroids
        else:
            self.dataset.centroids = centroids
            
        max_corr = np.zeros_like(self.dataset.vf_norm) - 1 # range from -1 to +1
        max_corr_idx = np.zeros_like(self.dataset.vf_norm, dtype=int) - 1 # -1 for background
        for cidx, centroid in enumerate(centroids):
            ctmap = calc_ctmap(centroid, normalized_vf, self.ncores)
            ctmap = np.nan_to_num(ctmap)
            mask = max_corr < ctmap
            max_corr[mask] = ctmap[mask]
            max_corr_idx[mask] = cidx
        self.dataset.max_correlations = max_corr
        self.dataset.celltype_maps = max_corr_idx
        return

    def filter_celltypemaps(self, min_r=0.6, min_norm=0.1, fill_blobs=True, min_blob_area=0, filter_params={}, output_mask=None):
        """
        Post-filter cell type maps created by `map_celltypes`.

        :param min_r: minimum threshold of the correlation.
        :type min_r: float
        :param min_norm: minimum threshold of the vector norm.
            If a string is given instead, then the threshold is automatically determined using
            sklearn's `threshold filter functions <https://scikit-image.org/docs/dev/api/skimage.filters.html>`_ (The functions start with `threshold_`).
        :type min_norm: str or float
        :param fill_blobs: If true, then the algorithm automatically fill holes in each blob.
        :type fill_blobs: bool
        :param min_blob_area: The blobs with its area less than this value will be removed.
        :type min_blob_area: int
        :param filter_params: Filter parameters used for the sklearn's threshold filter functions.
            Not used when `min_norm` is float.
        :type filter_params: dict
        :param output_mask: If given, the cell type maps will be filtered using the output mask.
        :type output_mask: np.ndarray(bool)
        """

        if isinstance(min_norm, str):
            # filter_params dict will be used for kwd params for filter_* functions.
            # some functions doesn't support param 'offset', therefore temporariliy remove it from here
            filter_offset = filter_params.pop('offset', 0)
        
        filtered_ctmaps = np.zeros_like(self.dataset.celltype_maps) - 1
        mask = np.zeros_like(self.dataset.vf_norm, dtype=bool)
        for cidx in range(len(self.dataset.centroids)):
            ctcorr = self.dataset.get_celltype_correlation(cidx)
            if isinstance(min_norm, str):
                for z in range(self.dataset.shape[2]):
                    if min_norm in ["local", "niblack", "sauvola", "localotsu"]:
                        im = np.zeros(self.dataset.vf_norm.shape[:-1])
                        im[ctcorr[..., z] > min_r] = self.dataset.vf_norm[..., z][ctcorr[..., z] > min_r]
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
                            highr_norm = self.dataset.vf_norm[..., z][ctcorr[..., z] > min_r]
                            #sigma = np.std(highr_norm)
                            if len(highr_norm) == 0 or np.max(highr_norm) == np.min(highr_norm):
                                min_norm_cut = np.max(self.dataset.vf_norm)
                            else:
                                min_norm_cut = filter_func(highr_norm, **filter_params)
                    min_norm_cut += filter_offset # manually apply filter offset
                    mask[..., z][np.logical_and(self.dataset.vf_norm[..., z] > min_norm_cut, ctcorr[..., z] > min_r)] = 1
            else:
                mask[np.logical_and(self.dataset.vf_norm > min_norm, ctcorr > min_r)] = 1

            if min_blob_area > 0 or fill_blobs:
                blob_labels = measure.label(mask, background=0)
                for bp in measure.regionprops(blob_labels):
                    if min_blob_area > 0 and bp.filled_area < min_blob_area:
                        for c in bp.coords:
                            mask[c[0], c[1], c[2]] = 0 # fill with zeros
                            #mask[c[0], c[1]] = 0 # fill with zeros
                        continue
                    if fill_blobs and bp.area != bp.filled_area:
                        minx, miny, minz, maxx, maxy, maxz = bp.bbox
                        mask[minx:maxx, miny:maxy, minz:maxz] |= bp.filled_image
                        #minr, minc, maxr, maxc = bp.bbox
                        #mask[minr:maxr, minc:maxc] |= bp.filled_image
                
            filtered_ctmaps[np.logical_and(mask == 1, np.logical_or(self.dataset.celltype_maps == -1, self.dataset.celltype_maps == cidx))] = cidx
        
        if isinstance(min_norm, str):
            # restore offset param
            filter_params['offset'] = filter_offset

        if output_mask is not None:
            filtered_ctmaps[~output_mask.astype(bool)] = -1
        self.dataset.filtered_celltype_maps = filtered_ctmaps
        
    def bin_celltypemaps(self, step=10, radius=100):
        """
        Sweep a sphere window along a lattice on the image, and count the number of cell types in each window.

        :param step: The lattice spacing.
        :type step: int
        :param radius: The radius of the sphere window.
        :type radius: int
        """
        def make_sphere_mask(radius):
            dia = radius*2+1
            X, Y, Z = np.ogrid[:dia, :dia, :dia]
            dist_from_center = np.sqrt((X - radius)**2 + (Y - radius)**2 + (Z - radius)**2)
            mask = dist_from_center <= radius
            return mask

        centers = np.array(self.dataset.vf_norm.shape) // 2
        steps = np.array(np.floor(centers / step) * 2 + np.array(self.dataset.vf_norm.shape) % 2, dtype=int)
        starts = centers - step * np.floor(centers / step)
        ends = starts + steps * step
        X, Y, Z = [np.arange(s, e, step, dtype=int) for s, e in zip(starts, ends)]

        ct_centers = np.zeros([len(X), len(Y), len(Z)], dtype=int)
        ct_counts = np.zeros([len(X), len(Y), len(Z), len(self.dataset.centroids)], dtype=int)

        ncelltypes = np.max(self.dataset.filtered_celltype_maps) + 1
        cnt_matrix = np.zeros([ncelltypes, ncelltypes])
        sphere_mask = make_sphere_mask(radius)

        for xidx, x in enumerate(X):
            for yidx, y in enumerate(Y):
                for zidx, z in enumerate(Z):
                    mask_slices = [slice(0, radius*2+1), slice(0, radius*2+1), slice(0, radius*2+1)]
                    s = [x - radius,     y - radius,     z - radius    ]
                    e = [x + radius + 1, y + radius + 1, z + radius + 1]

                    for ms_idx, ms in enumerate(s):
                        if ms < 0:
                            mask_slices[ms_idx] = slice(abs(ms), mask_slices[ms_idx].stop)
                            s[ms_idx] = 0
                    for me_idx, me in enumerate(e):
                        ctmap_size = self.dataset.filtered_celltype_maps.shape[me_idx]
                        #ctmap_size = 50
                        if me > ctmap_size:
                            mask_slices[me_idx] = slice(mask_slices[me_idx].start, (radius * 2 + 1) + ctmap_size - me)
                            e[me_idx] = ctmap_size

                    w = self.dataset.filtered_celltype_maps[s[0]:e[0],
                                                            s[1]:e[1],
                                                            s[2]:e[2]][sphere_mask[tuple(mask_slices)]] + 1

                    ct_centers[xidx, yidx, zidx] = self.dataset.filtered_celltype_maps[x, y, z]
                    ct_counts[xidx, yidx, zidx] = np.bincount(np.ravel(w), minlength=len(self.dataset.centroids) + 1)[1:]
                    
        self.dataset.celltype_binned_centers = ct_centers
        self.dataset.celltype_binned_counts = ct_counts
        return
        
    def find_domains(self, centroid_indices=[], n_clusters=10, norm_thres=0, merge_thres=0.6, merge_remote=True):
        """
        Find domains in the image, using the result of `bin_celltypemaps`.

        :param centroid_indices: The indices of centroids which will be used for determine tissue domains.
        :type centroid_indices: list(int)
        :param n_clusters: Initial number of clusters (domains) of agglomerative clustering.
        :type n_clusters: int
        :param norm_thres: Threshold for the total number of cell types in each window.
            The window which contains the number of cell-type pixels less than this value will be ignored.
        :type norm_thres: int
        :param merge_thres: Threshold for merging domains. The centroids of the domains
            which have higher correlation to this value will be merged.
        :type merge_thres: float
        :param merge_remote: If true, allow merging clusters that are not adjacent to each other.
        :type merge_remote: bool
        """
        def find_neighbors(m, l):
            neighbors = set()
            for x, y, z in zip(*np.where(m == l)):
                neighbors.add(m[x - 1, y    , z    ])
                neighbors.add(m[x + 1, y    , z    ])
                neighbors.add(m[x    , y - 1, z    ])
                neighbors.add(m[x    , y + 1, z    ])
                neighbors.add(m[x    , y    , z - 1])
                neighbors.add(m[x    , y    , z + 1])
                neighbors.add(m[x - 1, y - 1, z    ])
                neighbors.add(m[x + 1, y - 1, z    ])
                neighbors.add(m[x - 1, y + 1, z    ])
                neighbors.add(m[x + 1, y + 1, z    ])
                neighbors.add(m[x - 1, y    , z - 1])
                neighbors.add(m[x + 1, y    , z - 1])
                neighbors.add(m[x - 1, y    , z + 1])
                neighbors.add(m[x + 1, y    , z + 1])
                neighbors.add(m[x    , y - 1, z - 1])
                neighbors.add(m[x    , y + 1, z - 1])
                neighbors.add(m[x    , y - 1, z + 1])
                neighbors.add(m[x    , y + 1, z + 1])
                neighbors.add(m[x - 1, y - 1, z - 1])
                neighbors.add(m[x + 1, y - 1, z - 1])
                neighbors.add(m[x - 1, y - 1, z + 1])
                neighbors.add(m[x + 1, y - 1, z + 1])
                neighbors.add(m[x - 1, y + 1, z - 1])
                neighbors.add(m[x + 1, y + 1, z - 1])
                neighbors.add(m[x - 1, y + 1, z + 1])
                neighbors.add(m[x + 1, y + 1, z + 1])
            return neighbors
        
        if self.dataset.celltype_binned_counts is None:
            raise AssertionError("Run 'bin_celltypemap()' method first!")

        if len(centroid_indices) > 0:
            binned_ctmaps = self.dataset.celltype_binned_counts[..., centroid_indices]
        else:
            binned_ctmaps = self.dataset.celltype_binned_counts

        binned_ctmaps_norm = np.sum(binned_ctmaps, axis=3)

        ctvf_vecs = binned_ctmaps[binned_ctmaps_norm > norm_thres]
        ctvf_vecs_normalized = preprocessing.normalize(ctvf_vecs, norm='l1', axis=1)

        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', affinity='euclidean').fit(ctvf_vecs_normalized)
        labels_predicted = clustering.labels_ + 1
        
        layer_map = np.zeros(binned_ctmaps_norm.shape)
        layer_map[binned_ctmaps_norm > norm_thres] = labels_predicted
        layer_map = measure.label(layer_map)
        
        if merge_thres < 1.0:
            while True:
                uniq_labels = np.array(list(set(list(np.ravel(layer_map))) - set([0])))
                if not merge_remote:
                    layer_map_padded = np.pad(layer_map, 1, mode='constant', constant_values=0)
                    neighbors_dic = {}
                    for lbl in uniq_labels:
                        neighbors_dic[lbl] = find_neighbors(layer_map_padded, lbl)
                cluster_centroids = []
                for lbl in uniq_labels:
                    cluster_centroids.append(np.mean(binned_ctmaps[layer_map == lbl], axis=0))
                max_corr = 0
                #max_corr_indices = (0, 0, )
                for i in range(len(uniq_labels)):
                    for j in range(i+1, len(uniq_labels)):
                        lbl_i, lbl_j = uniq_labels[i], uniq_labels[j]
                        if lbl_i == 0 or lbl_j == 0:
                            continue
                        corr_ij = corr(cluster_centroids[i], cluster_centroids[j])
                        if corr_ij > max_corr and (merge_remote or lbl_j in neighbors_dic[lbl_i]):
                            max_corr = corr_ij
                            max_corr_indices = (lbl_i, lbl_j, )
                if max_corr > merge_thres:
                    layer_map[layer_map == max_corr_indices[1]] = max_corr_indices[0]
                else:
                    break

        """
        if min_size > 0:
            labeled_layer_map = measure.label(layer_map)
            labeled_layer_map_padded = np.pad(labeled_layer_map, 1, mode='constant', constant_values=0)
            for prop in measure.regionprops(labeled_layer_map):
                if prop.area < min_size:
                    find_neighbors(layer_map_padded, )
        """

        uniq_labels = sorted(set(list(np.ravel(layer_map))) - set([0]))
        for i, lbl in enumerate(uniq_labels, start=1):
            layer_map[layer_map == lbl] = i
        
        resized_layer_map = zoom(layer_map, np.array(self.dataset.vf_norm.shape)/np.array(layer_map.shape), order=0) - 1
        resized_layer_map2 = np.array(resized_layer_map, copy=True)
        resized_layer_map2[self.dataset.filtered_celltype_maps == -1] = -1
        
        self.dataset.inferred_domains = resized_layer_map
        self.dataset.inferred_domains_cells = resized_layer_map2
     
    def exclude_and_merge_domains(self, exclude=[], merge=[]):
        """
        Manually exclude or merge domains.

        :param exclude: Indices of the domains which will be excluded.
        :type exclude: list(int)
        :param merge: List of indices of the domains which will be merged.
        :type merge: list(list(int))
        """
        for i in exclude:
            self.dataset.inferred_domains[self.dataset.inferred_domains == i] = -1
            self.dataset.inferred_domains_cells[self.dataset.inferred_domains_cells == i] = -1
            
        for i in merge:
            for j in i[1:]:
                self.dataset.inferred_domains[self.dataset.inferred_domains == j] = i[0]
                self.dataset.inferred_domains_cells[self.dataset.inferred_domains_cells == j] = i[0]

        uniq_indices = np.unique(self.dataset.inferred_domains_cells)
        if -1 in uniq_indices:
            uniq_indices = uniq_indices[1:]
            
        for new_idx, i in enumerate(uniq_indices):
            self.dataset.inferred_domains[self.dataset.inferred_domains == i] = new_idx
            self.dataset.inferred_domains_cells[self.dataset.inferred_domains_cells == i] = new_idx

    def calc_cell_type_compositions(self):
        """
        Calculate cell type compositions in each domain.
        """
        cell_type_compositions = []
        for i in range(np.max(self.dataset.inferred_domains) + 1):
            counts = np.bincount(self.dataset.filtered_celltype_maps[self.dataset.inferred_domains == i] + 1, minlength=len(self.dataset.centroids) + 1)
            cell_type_compositions.append(counts[1:])
        
        masked_ctmap = self.dataset.filtered_celltype_maps[self.dataset.filtered_celltype_maps != -1]
        counts_all = np.array(np.bincount(masked_ctmap, minlength=len(self.dataset.centroids)), dtype=float)
        cell_type_compositions.append(counts_all) # Add proportion from the whole tissue
        cell_type_compositions = preprocessing.normalize(cell_type_compositions, axis=1, norm='l1')
        self.dataset.inferred_domains_compositions = cell_type_compositions
        
        
    def calc_spatial_relationship(self):
        """
        Calculate spatial relationship between the domains using the result of `bin_celltypemap`.
        """
        if self.dataset.celltype_binned_counts is None:
            raise AssertionError("Run 'bin_celltypemap()' method first!")
            
        ct_centers = self.dataset.celltype_binned_centers
        
        sparel = np.zeros([len(self.dataset.centroids), len(self.dataset.centroids)])
        for idx in np.unique(ct_centers):
            sparel[idx, :] = np.sum(self.dataset.celltype_binned_counts[ct_centers == idx], axis=0)

        self.dataset.spatial_relationships = preprocessing.normalize(sparel, axis=1, norm='l1')
