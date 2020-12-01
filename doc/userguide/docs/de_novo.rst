SSAM *de novo* analysis
=======================

While we believe that the `guided mode of SSAM <guided.md>`__ to be able
to generate good cell-type maps rapidly, the *de novo mode* provide much
more accurate results.

The steps of the *de novo* analysis are briefly discussed below, with
links to more detailed discussion:

-  `setting cell-type map correlation
   threshold <docs/celltype_map_thresh_d.md>`__
-  `visualisation of cell-type signatures: heatmap, tSNE,
   UMAP <docs/visualisation.md>`__

Clustering of expression vectors
--------------------------------

Once the local maxima have been selected and
`filtered <max_filtering.md>`__, we can perform `clustering
analysis <clustering.md>`__. SSAM supports `a number of clustering
methods <clustering.md>`__. Here we use the Louvain algorithm using 22
principle components, a resolution of 0.15.

::

   analysis.cluster_vectors(
       method="louvain", 
       min_cluster_size=0,
       pca_dims=22,
       resolution=0.15,
       metric='correlation')

Remove outliers
---------------

In order to improve the representation of the cluster centroid for the
entire cluster we `filter local maxima
outliers <clustering.md#removing-outliers>`__ of the clusters when they
have lower than 0.6 correlation to the cluster medoid:

::

   analysis.filter_celltypemaps(min_norm=filter_method, filter_params=filter_params, min_r=0.6, output_mask=output_mask)

Cluster annotation and diagnostics
----------------------------------

SSAM provides `diagnostic plots <diagnostics.md>`__ which can be used to
evaluate the quality of clusters, and `facilitates the annotation of
clusters <cluster_annotation.md>`__.

Visualisng the clusters
-----------------------

SSAM supports `cluster visualisation via heatmaps, and 2D embedding
(t-SNE and UMAP) <visualisation.md>`__. Here we give an example of the
t-SNE plot:

::

   plt.figure(figsize=[5, 5])
   ds.plot_tsne(pca_dims=22, metric="correlation", s=5, run_tsne=True)

|image0|

Cell type map
-------------

Once the clusters have been evaluated for quality, we can generate the
*de novo* cell-type map. This involves `classifying all the pixels in
the tissue image based on a correlation
threshold <celltype_map_thresh_d.md>`__, which in this case is ``0.6``:

::

   analysis.map_celltypes()
   analysis.filter_celltypemaps(min_norm=filter_method, filter_params=filter_params, min_r=0.6, fill_blobs=True, min_blob_area=50, output_mask=output_mask)

|image1|

.. |image0| image:: ../images/tsne.png
.. |image1| image:: ../images/de_novo.png

