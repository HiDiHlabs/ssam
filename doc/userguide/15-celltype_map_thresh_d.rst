Thresholding the de-novo cell-type map
======================================

After cell-type signatures are calculated, the tissue image can be
classified. The classification of each pixel is based on the Pearson
correlation metric (although an `experimental adversarial autoencoder
based classification method <aaec.md>`__ can be applied).

We found that a minimum correlation threshold (``min_r``) of 0.3 worked
well for guided mode based on single cell RNAseq cell-type signatures,
and 0.6 worked well for *de novo* mode.

Below we show how the cell-type map changes using correlation thresholds
of ``0.4,0.6,0.8`` for the guided cell-type map.

::

   analysis.map_celltypes()

   analysis.filter_celltypemaps(min_norm=filter_method, filter_params=filter_params, min_r=0.4, fill_blobs=True, min_blob_area=50, output_mask=output_mask)
   plt.figure(figsize=[5, 5])
   ds.plot_celltypes_map(colors=denovo_celltype_colors, rotate=1, set_alpha=False)

   analysis.filter_celltypemaps(min_norm=filter_method, filter_params=filter_params, min_r=0.6, fill_blobs=True, min_blob_area=50, output_mask=output_mask)
   plt.figure(figsize=[5, 5])
   ds.plot_celltypes_map(colors=denovo_celltype_colors, rotate=1, set_alpha=False)

   analysis.filter_celltypemaps(min_norm=filter_method, filter_params=filter_params, min_r=0.8, fill_blobs=True, min_blob_area=50, output_mask=output_mask)
   plt.figure(figsize=[5, 5])
   ds.plot_celltypes_map(colors=denovo_celltype_colors, rotate=1, set_alpha=False)
