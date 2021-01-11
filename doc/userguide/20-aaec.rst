Cell-type classification using Adversarial Autoencoders
=======================================================

The default classification algorithm is based on Pearson correlation as
this has been `shown to be effective for automatic classification of
cell types <https://doi.org/10.1186/s13059-019-1795-z>`__ for single
cell RNAseq experiments. This proved to be both highly performant and
accurate also for spatial gene expression data. However, it may be
desirable to explore other classification methods.

One recent and exciting Deep Learning framework that achieve competitive
results in generative modeling and semi-supervised classification tasks
are `adversarial autoencoders <https://arxiv.org/abs/1511.05644>`__.

SSAM implements a modified version of adversarial autoencoder classifier
based on the `original
implementation <https://github.com/shaharazulay/adversarial-autoencoder-classifier>`__
by `Shahar Azulay <https://github.com/shaharazulay>`__.

Mapping cell types using an adversarial autoencoder
---------------------------------------------------

In order to use the AAEC classification of pixels instead of the Pearson
correlation based method, simply replace ``analysis.map_celltypes()``
with :

::

   analysis.map_celltypes_aaec(epochs=1000, seed=0, batch_size=1000, chunk_size=100000, z_dim=10, noise=0)

