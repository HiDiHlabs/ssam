The shape of the kernel
=======================

The shape of the kernel is defined by the `kernel
function <https://en.wikipedia.org/wiki/Kernel_(statistics)>`__. The
shape of the kernel determines how the mRNA signal is smoothed.

We adopt the use of the Gaussian kernel due to it’s popular use in
signal processing, however other kernel functions can be used: - we have
had success in using semi-circle kernels when applied to `ISS data of
the human pancreas <https://doi.org/10.1053/j.gastro.2020.11.010>`__ -
the `Epanechnikov kernel <https://doi.org/10.1137%2F1114019>`__
minimizes AMISE and has therefore been described as optimal

The following exmaples shows how you can apply a semicircular kernel
instead of a Gaussian.

::

   # code to change the shape of the kernel (@sebastiantiesmeyer)
