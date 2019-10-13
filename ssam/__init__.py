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
from sklearn.neighbors.kde import KernelDensity
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

from .utils import corr, calc_ctmap, calc_corrmap, flood_fill, calc_kde

def calc_slice(args):
    
    (gidx, maxdist, bandwidth, save_dir, gene_name, shape, locations, re_run, sampling_distance) = args
        
    #bandwidth = 1#bandwidth**0.5
    span = np.linspace(-maxdist,maxdist,maxdist*2+1)
    X,Y = np.meshgrid(span,span)#,[0])
    
    def create_kernel(x,y,z,X=X,Y=Y,span=span):#,Z=Z):
        X_=(-x+X)/bandwidth
        Y_=(-y+Y)/bandwidth
#        Z_=(z-Z)/bandwidth        
        return np.exp(-0.5*(X_**2+Y_**2))#+Z_**2))  
    
    print('Processing gene %s'%gene_name, gidx )

    pdf_filename = os.path.join(save_dir, 'pdf_sd%s_bw%s_%s.npy'%(
        ('%f' % sampling_distance).rstrip('0').rstrip('.'),
        ('%f' % bandwidth).rstrip('0').rstrip('.'),
        gene_name)
    )            
      
    if not os.path.exists(pdf_filename) or re_run:  
       
        vf_slice = np.zeros([s+2*maxdist for s in shape[:2]])
        
        for n_dp,[x,y,z] in enumerate(locations):
            int_x,int_y = (int(x)+maxdist,int(y)+maxdist)
            rem_x,rem_y = (x%1,y%1)
            
            kernel = create_kernel(rem_x,rem_y,0)
                
            x_ = int_x-maxdist
            y_ = int_y-maxdist
            _x = int_x+maxdist+1
            _y = int_y+maxdist+1
            
            vf_slice[x_:_x,y_:_y]+=kernel.squeeze().T
        
        pdf = vf_slice[maxdist:-maxdist,maxdist:-maxdist]
        
        pdf/=pdf.sum()
        
        np.save(pdf_filename, pdf)
    

def run_sctransform(data, **kwargs):
    vst_options = ['%s = "%s"'%(k, v) if type(v) is str else '%s = %s'%(k, v) for k, v in kwargs.items()]
    if len(vst_options) == 0:
        vst_opt_str = ''
    else:
        vst_opt_str = ', ' + ', '.join(vst_options)
    with TemporaryDirectory() as tmpdirname:
        ifn, ofn, pfn, rfn = [os.path.join(tmpdirname, e) for e in ["in.feather", "out.feather", "fit_params.feather", "script.R"]]
        df = pd.DataFrame(data, columns=[str(e) for e in range(data.shape[1])])
        df.to_feather(ifn)
        rcmd = 'library(feather); library(sctransform); mat <- t(as.matrix(read_feather("{0}"))); colnames(mat) <- 1:ncol(mat); res <- sctransform::vst(mat{1}); write_feather(as.data.frame(t(res$y)), "{2}"); write_feather(as.data.frame(res$model_pars_fit), "{3}");'.format(ifn, vst_opt_str, ofn, pfn)
        with open(rfn, "w") as f:
            f.write(rcmd)
        subprocess.check_output("Rscript " + rfn, shell=True)
        return pd.read_feather(ofn), pd.read_feather(pfn)

class SSAMDataset(object):
    def __init__(self, genes, locations, width, height, depth=1):
        """
        SSAMDataset(genes, locations, width, height, depth = 1, ncores = -1, save_dir = "", verbose = False)
            A class to store intial values and results of SSAM analysis.

            Parameters
            ----------
            genes : list(string)
                The genes that will be used for the analysis.
            locations : list(numpy.ndarray)
                Location of the mRNAs in um, given as a list of
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
    
    def get_celltype_correlation(self, idx):
        rtn = np.zeros_like(self.max_correlations) - 1
        rtn[self.celltype_maps == idx] = self.max_correlations[self.celltype_maps == idx]
        return rtn
    
    def plot_l1norm(self, cmap="viridis", rotate=0, z=None):
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
        
    def plot_tsne(self, run_tsne=False, pca_dims=10, n_iter=5000, perplexity=70, early_exaggeration=10, metric="correlation", exclude_bad_clusters=True, s=None, random_state=0, colors=[], excluded_color="#00000033", cmap="jet", tsne_kwargs={}):
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
        plt.imshow(self.expanded_mask, vmin=0, vmax=1, cmap=cmap)
        return
    
    def plot_correlation_map(self, cmap='hot'): # TODO
        plt.imshow(self.corr_map, vmin=0.995, vmax=1.0, cmap=cmap)
        plt.colorbar()
        return
    
    def plot_celltypes_map(self, background="black", centroid_indices=[], colors=None, cmap='jet', rotate=0, min_r=0.6, set_alpha=True, z=None):
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
    
    def plot_diagnostic_plot(self, centroid_index, cluster_name=None, cluster_color=None, cmap=None, use_embedding="tsne", known_signatures=[], correlation_methods=[]):
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
        self.plot_l1norm(rotate=1, cmap="Greys")

        ax = plt.subplot(1, 4, 2)
        ctmap = np.zeros([self.filtered_celltype_maps.shape[1], self.filtered_celltype_maps.shape[0], 4])
        ctmap[self.filtered_celltype_maps[..., 0].T == centroid_index] = to_rgba(cluster_color)
        ctmap[np.logical_and(self.filtered_celltype_maps[..., 0].T != centroid_index, self.filtered_celltype_maps[..., 0].T > -1)] = [0.9, 0.9, 0.9, 1]
        ax.imshow(ctmap)
        plt.xlim([self.celltype_maps.shape[0], 0])

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
        
    def plot_celltype_composition(self, domain_index, cell_type_colors=None, cell_type_cmap='jet', cell_type_orders=None, label_cutoff=0.03, pctdistance=1.15, **kwarg):
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
                pctdistance=pctdistance)

    def plot_spatial_relationships(self, cluster_labels, *args, **kwargs):
        sns.heatmap(self.spatial_relationships, *args, xticklabels=cluster_labels, yticklabels=cluster_labels, **kwargs)    

        
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
                Set it 2.5 to make FWTM of Gaussian kernel to be ~10um (assume that avg. cell diameter is 10um).
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
   
    def run_kde_fast(self, kernel='gaussian', bandwidth=2.0, sampling_distance=1.0, re_run=False, n_cores=5):
        '''
            kernel : string, optional
                Kernel for density estimation.
            bandwidth : float, optional
                Parameter to adjust width of kernel.
                Set it 2.5 to make FWTM of Gaussian kernel to be ~10um (assume that avg. cell diameter is 10um).
            sampling_distance : float, optional
                Grid spacing in um.
        '''

        maxdist = int(bandwidth*4)
        with closing(Pool(n_cores, maxtasksperchild=1)) as p:
            idcs = np.argsort([len(i) for i in self.dataset.locations])[::-1]
            self.dataset.vf = np.zeros(self.dataset.shape[:2]+(len(idcs),))
            p.map(calc_slice,[(gidx, maxdist, bandwidth, self.save_dir, 
                                   self.dataset.genes[gidx], self.dataset.shape, 
                                   self.dataset.locations[gidx], re_run,
                                   sampling_distance) for gidx in idcs])
            p.close()
            p.join()
        
        for gidx,gene in enumerate(self.dataset.genes):
            pdf_filename = os.path.join(self.save_dir, 'pdf_sd%s_bw%s_%s.npy'%(
                ('%f' % sampling_distance).rstrip('0').rstrip('.'),
                ('%f' % bandwidth).rstrip('0').rstrip('.'),
                gene)
            )
            self.dataset.vf[...,gidx] = np.load(pdf_filename)
        

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
    
    def find_localmax(self, search_size=21, min_norm=0, min_expression=0.05, mask=None):
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
            min_expression : float, optional
                Minimum value of gene expression in a unit pixel at the local maxima.
            mask: numpy.ndarray, optional
                If given, find vectors in the masked region, instead of the whole image.
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
    
    def normalize_vectors_sctransform(self, use_expanded_vectors=False, normalize_vf=True):
        """
        nomalize_vectors_rnb(use_expanded_vectors = False, normalize_gene = False, normalize_vector = True, log_transform = True, scale = True)
            Normalize and regularize vectors using SCtransform
        """
        if use_expanded_vectors:
            vec = np.array(self.dataset.expanded_vectors, copy=True)
        else:
            vec = np.array(self.dataset.vf[self.dataset.local_maxs], copy=True)
        
        norm_vec, fit_params = run_sctransform(vec)
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
        nomalize_vectors(use_expanded_vectors = False, normalize_gene = False, normalize_vector = True, log_transform = True, scale = True)
            Normalize and regularize vectors
            
            Parameters
            ----------
            use_expanded_vectors : bool (default: False)
                If True, use averaged vectors nearby local maxima of the vector field.
            normalize_gene: bool (default: True)
                If True, normalize vectors by sum of each gene expression across all vectors.
            normalize_vector: bool (default: True)
                If True, normalize vectors by sum of all gene expression of each vector.
            log_transform: bool (default: True)
                If True, vectors are log transformed.
            scale: bool (default: True)
                If True, vectors are z-scaled (mean centered and scaled by stdev).
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

    def cluster_vectors(self, pca_dims=10, min_cluster_size=0, resolution=0.6, prune=1.0/15.0, snn_neighbors=30, max_correlation=0.89, metric="euclidean", subclustering=False, centroid_correction_threshold=0.8, random_state=0):
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
        vecs_normalized_dimreduced = PCA(n_components=pca_dims, random_state=random_state).fit_transform(vecs_normalized)

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
                
        new_labels = self.__correct_cluster_labels(all_lbls, centroid_correction_threshold)
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
        exclude_and_merge_clusters(exclude, merge)
            Exclude bad clusters (including the vectors in the clusters), and merge similar clusters for the downstream analysis.

            Parameters
            ----------
            exclude: list(int)
                List of cluster indices to be excluded.
            merge: np.array(np.array(int)) or list(list(int))
                List of list of cluster indices to be merged.
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
        map_celltypes(centroids = None)
            Create correlation maps between the centroids and the vector field.
            Each correlation map corresponds each cell type's image map.

            Parameters
            ----------
            centroids: np.array(float) or list(np.array(float)), default: None
                If given, map celltypes with the given cluster centroids. Ignore 'use_medoids' parameter.
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

    def filter_celltypemaps(self, min_r=0.6, min_norm=0.1, fill_blobs=True, min_blob_area=100, filter_params={}, output_mask=None):
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
        if self.dataset.celltype_binned_counts is None:
            raise AssertionError("Run 'bin_celltypemap()' method first!")
            
        ct_centers = self.dataset.celltype_binned_centers
        
        sparel = np.zeros([len(self.dataset.centroids), len(self.dataset.centroids)])
        for idx in np.unique(ct_centers):
            sparel[idx, :] = np.sum(self.dataset.celltype_binned_counts[ct_centers == idx], axis=0)

        self.dataset.spatial_relationships = preprocessing.normalize(sparel, axis=1, norm='l1')
        
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
    analysis.run_kde_fast()
    analysis.find_localmax()
    analysis.expand_localmax()
