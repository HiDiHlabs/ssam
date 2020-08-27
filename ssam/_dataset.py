import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
import multiprocessing
import sys, os
sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from tempfile import mkdtemp
from matplotlib.colors import ListedColormap
from .utils import corr

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
    :param save_dir: Directory to store intermediate data as zarr groups (e.g. density / vector field).
        Any data which already exists will be loaded and reused.
    :type save_dir: str
    """
        
    def __init__(self, save_dir="", overwrite=False):
        self._vf = None
        self._vf_norm = None
        self._local_maxs = None
        self._selected_vectors = None
        self.normalized_vectors = None
        self.expanded_vectors = None
        self.cluster_labels = None
        #self.corr_map = None
        self.tsne = None
        self.umap = None
        self.vf_normalized = None
        self.excluded_clusters = None
        self.celltype_binned_counts = None
        self.max_probabilities = None
        if len(save_dir) == 0:
            save_dir = mkdtemp()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
    
    @property
    def local_maxs(self):
        return self._local_maxs
    
    @local_maxs.setter
    def local_maxs(self, v):
        self._local_maxs = v
        self._selected_vectors = None
    
    @property
    def selected_vectors(self):
        if self._selected_vectors is None:
            assert self._local_maxs is not None
            mask = np.zeros(self.vf_norm.shape, dtype=bool)
            mask[self.local_maxs] = True
            self._selected_vectors = self.vf.reshape([-1, len(self.genes)])[np.ravel(mask)].compute()
        return self._selected_vectors
    
    @selected_vectors.setter
    def selected_vectors(self, v):
        self._selected_vectors = v

    @property
    def vf(self):
        """
        Vector field as a numpy.ndarray.
        """
        return self._vf
    
    @vf.setter
    def vf(self, vf):
        self._vf = vf
        self._vf_norm = None
        
    @property
    def vf_norm(self):
        """
        `L1-norm <http://mathworld.wolfram.com/L1-Norm.html>`_ of the vector field as a numpy.ndarray.
        """

        if self.vf is None:
            return None
        if self._vf_norm is None:
            self._vf_norm = self.vf.sum(axis=3).persist()
        return self._vf_norm
    
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
        
    def _run_pca(self, exclude_bad_clusters, pca_dims, random_state):
        if exclude_bad_clusters:
            good_vecs = self.normalized_vectors[self.filtered_cluster_labels != -1, :]
        else:
            good_vecs = self.normalized_vectors
        if pca_dims > 0:
            return PCA(n_components=pca_dims, random_state=random_state).fit_transform(good_vecs)
        else:
            return good_vecs
        
    def plot_tsne(self, re_run=True, pca_dims=-1, n_iter=5000, perplexity=70, early_exaggeration=10,
                  metric="correlation", exclude_bad_clusters=True, s=None, random_state=0, colors=[], excluded_color="#00000033", cmap="jet", tsne_kwargs={}):
        """
        Scatter plot the tSNE embedding.

        :param re_run: If false, loads precomputed tSNE result.
        :type re_run: bool
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
        if re_run or self.tsne is None:
            pcs = self._run_pca(exclude_bad_clusters, pca_dims, random_state)
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

    def plot_umap(self, re_run=True, pca_dims=-1, metric="correlation", min_dist=0.8, exclude_bad_clusters=True, s=None, random_state=0, colors=[], excluded_color="#00000033", cmap="jet", umap_kwargs={}):
        """
        Scatter plot the UMAP embedding.

        :param re_run: If false, loads precomputed UMAP result.
        :type re_run: bool
        :param pca_dims: Number of PCA dimensions used for the UMAP embedding.
        :type pca_dims: int
        :param metric: Metric for calculation of distance between vectors in gene expression space.
        :type metric: str
        :param min_dist: 'min_dist' parameter for UMAP.
        :type min_dist: float
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
        if re_run or self.umap is None:
            pcs = self._run_pca(exclude_bad_clusters, pca_dims, random_state)
            self.umap = UMAP(metric=metric, random_state=random_state, min_dist=min_dist, **umap_kwargs).fit_transform(pcs[:, :pca_dims])
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
        ctmap = np.zeros([self.filtered_celltype_maps.shape[1], self.filtered_celltype_maps.shape[0], 4])
        ctmap[self.filtered_celltype_maps[..., z].T == centroid_index] = to_rgba(cluster_color)
        ctmap[np.logical_and(self.filtered_celltype_maps[..., z].T != centroid_index, self.filtered_celltype_maps[..., 0].T > -1)] = [0.9, 0.9, 0.9, 1]
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
        Get maximum correlation values of a cell type map between the given cluster's centroid to the vector field.
        
        :param idx: Index of a cluster
        :type idx: int
        :return: Correlation values of a cell type map of the specified cluster's centroid
        :rtype: numpy.ndarray
        """
        rtn = np.zeros_like(self.max_correlations) - 1
        rtn[self.celltype_maps == idx] = self.max_correlations[self.celltype_maps == idx]
        return rtn
    
    def get_celltype_probability(self, idx):
        """
        Get maximum probability map of a cell type.
        
        :param idx: Index of a cluster
        :type idx: int
        :return: Maximum probability map
        :rtype: numpy.ndarray
        """
        rtn = np.zeros_like(self.max_probabilities) - 1
        rtn[self.celltype_maps == idx] = self.max_probabilities[self.celltype_maps == idx]
        return rtn