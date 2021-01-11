Installation
============

A step-by-step guide
--------------------

The easiest way to prepare a python environment for SSAM is using
`conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`__.
Keeping python projects in isolated environments prevents dependency
version conflicts or conflicts with your OS installation of python which
usually depends on older versions incompatible with current scientific
packages.

Create your environment:

::

   conda create -n ssam python=3.6

Remember to activate before using it:

::

   conda activate ssam

Now we use conda to install some dependencies into our ssam environment:

::

   conda install gxx_linux-64=7.3.0 numpy=1.19.2 pip R=3.6 pyarrow=0.15.1

Now we can install the R packages ``sctransform`` and ``feather``. Open
R and type:

::

   install.packages("sctransform")
   install.packages("feather")

Finally we switch to pip:

.. raw:: html

   <!--
   ```
   pip install ssam
   ```
   -->

::

   pip install git+https://github.com/HiDiHlabs/ssam.git

Next we can download and prepare our `data <data.md>`__.

SSAM’s source code
------------------

In case you want to work with `SSAM’s source
code <https://github.com/HiDiHlabs/ssam>`__, it is also hosted on github.
