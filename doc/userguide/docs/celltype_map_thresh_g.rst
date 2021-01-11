Thresholding the guided cell-type map
=====================================

After cell-type signatures are provided, the tissue image can be
classified. The classification of each pixel is based on the Pearson
correlation metric (although an `experimental adversarial autoencoder
based classification method <aaec.md>`__ can be applied).

We found that a minimum correlation threshold (``min_r``) of 0.3 worked
well for guided mode based on single cell RNAseq cell-type signatures,
and 0.6 worked well for *de novo* mode.

Below we show how the cell-type map changes using correlation thresholds
of ``0.15,0.3,0.45`` using the scRNAseq signatures

::

   scrna_uniq_labels = [scrna_cl_metadata_dic[i][0] for i in scrna_uniq_clusters]
   scrna_colors = [scrna_cl_metadata_dic[i][1] for i in scrna_uniq_clusters]

   analysis.map_celltypes(scrna_centroids)

   analysis.filter_celltypemaps(min_norm=filter_method, filter_params=filter_params, min_r=0.15, output_mask=output_mask) # post-filter cell-
   plt.figure(figsize=[5, 5])
   ds.plot_celltypes_map(rotate=1, colors=scrna_colors, set_alpha=False)

   analysis.filter_celltypemaps(min_norm=filter_method, filter_params=filter_params, min_r=0.3, output_mask=output_mask) # post-filter cell-
   plt.figure(figsize=[5, 5])
   ds.plot_celltypes_map(rotate=1, colors=scrna_colors, set_alpha=False)

   analysis.filter_celltypemaps(min_norm=filter_method, filter_params=filter_params, min_r=0.45, output_mask=output_mask) # post-filter cell-
   plt.figure(figsize=[5, 5])
   ds.plot_celltypes_map(rotate=1, colors=scrna_colors, set_alpha=False)
