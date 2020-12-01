Clustering Local L-1 Maxima
===========================

In the *de novo* mode analysis, after the local maxima have been
identified from the tissue image, they are clustered.

The default clustering algorithm is based on `Louvain community
detection <https://doi.org/10.1088%2F1742-5468%2F2008%2F10%2FP10008>`__.
SSAM also supports clustering using ``hdbscan`` and ``optics``.

It can be initiated by:

::

   analysis.cluster_vectors(method="louvain", 
                            pca_dims=-1, 
                            min_cluster_size=2, 
                            max_correlation=1.0, 
                            metric="correlation",
                            outlier_detection_method='medoid-correlation', 
                            outlier_detection_kwargs={}, 
                            random_state=0, 
                            **kwargs)

… where - ``method`` can be ``louvain``, ``hdbscan``, ``optics``. -
``pca_dims`` are the number of principal componants used for clustering.
- ``min_cluster_size`` is the minimum cluster size. - ``resolution`` is
the resolution for Louvain community detection. - ``prune`` is the
threshold for Jaccard index (weight of SNN network). If it is smaller
than prune, it is set to zero. - ``snn_neighbors`` is the number of
neighbors for SNN network. - ``max_correlation`` is the threshold for
which clusters with higher correlation to this value will be merged. -
``metric`` is the metric for calculation of distance between vectors in
gene expression space. - ``subclustering`` if set to True, each cluster
will be clustered once again with DBSCAN algorithm to find more
subclusters. - ``dbscan_eps`` is the ``eps`` value for DBSCAN
subclustering. Not used when ‘subclustering’ is set False. -
``centroid_correction_threshold`` is the threshold for which centroid
will be recalculated with the vectors which have the correlation to the
cluster medoid equal or higher than this value. - ``random_state`` is
the random seed or scikit-learn’s random state object to replicate the
same result

Removing outliers
-----------------

The cell type signature is determined as the centroid of the cluster.
This can be affected by outliers, so SSAM supports a number of outlier
removal methods:

::

   analysis.remove_outliers(outlier_detection_method='medoid-correlation', outlier_detection_kwargs={}, normalize=True)

.. where - ``outlier_detection_method`` can be ``medoid-correlation``,
``robust-covariance``, ``one-class-svm``, ``isolation-forest``,
``local-outlier-factor`` - ``outlier_detection_kwargs`` are arguments
passed to the outlier detection method
