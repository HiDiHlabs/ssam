Segmenting the SSAM cell type map
=================================

While we demonstrate the accuracy of SSAM in reconstructing celltype
maps, we understand that many applications in biology require cell
segmentation. As such, the development branch of SSAM supports
segmentation of the celltype map using the ``watershed`` algorithm.

**This is an experimental feature!**

The segmentation of the cell type map can be performed by:

.. code-block:: python

   # Load DAPI image
   with open('zenodo/osmFISH/raw_data/im_nuc_small.pickle', 'rb') as f:
       dapi = pickle.load(f)
   dapi_small = np.hstack([dapi.T[:1640], np.zeros([1640, 12])]).reshape(ds.vf_norm.shape)
   
   # Threshold DAPI image to create markers
   dapi_threshold = filters.threshold_local(dapi_small[..., 0], 35, offset=-0.0002)
   dapi_thresh_im = dapi_small[..., 0] > dapi_threshold
   dapi_thresh_im = dapi_thresh_im.reshape(ds.vf_norm.shape).astype(np.uint8) * 255
   
   # Run watershed segmentation of cell-type maps with DAPI as markers
   # After running below, the segmentation data will be available as:
   #  - Segmentations: ds.watershed_segmentations
   #  - Cell-type map: ds.watershed_celltype_map
   analysis.run_watershed(dapi_thresh_im)

Below we demonstrate the application of the segmentation on the *de
novo* celltype map generated for the mouse SSp osmFISH data.

|image0|

.. |image0| image:: ../images/segmented_celltype_map.png

