Data Preparation
================

Download VISp data
------------------

In this tutorial we work with data of the murine primary visual cortex
(VISp) profiled using multiplexed smFISH. Further details are available
in the SSAM publication (Park, et. al. 2019).

First, download the data and unpack it:

::

   curl "https://zenodo.org/record/3478502/files/supplemental_data_ssam_2019.zip?download=1" -o zenodo.zip
   unzip zenodo.zip

Load data into python
---------------------

Let’s start with loading our python packages:

::

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import ssam

Now we can load the mRNA spot table. Each row describes one mRNA spot
and the columns contain its coordinates and target gene. We load the
required columns into a dataframe:

::

   df = pd.read_csv(
       "zenodo/multiplexed_smFISH/raw_data/smFISH_MCT_CZI_Panel_0_spot_table.csv",
       usecols=['x', 'y', 'z', 'target'])

If your dataset is organized differently, you will have to reshape it
before continuing with the next steps. ## Transform Data

Because SSAM analysis is rooted in a cellular scale we transform the
coordinates from a laboratory system into micrometers. Also we make them
a bit tidier:

::

   um_per_pixel = 0.1

   df.x = (df.x - df.x.min()) * um_per_pixel + 10
   df.y = (df.y - df.y.min()) * um_per_pixel + 10
   df.z = (df.z - df.z.min()) * um_per_pixel + 10

Prepare data for SSAM
---------------------

To create a ``SSAMDataset`` object we need to provide four arguments: -
a list of gene names profiled in the experiment: ``genes`` - a list of
lists that contains the coordinates of each gene: ``coord_list`` - the
``width`` of the image - the ``height`` of the image

The width and height are straightforward to infer from the dimensions of
the image:

::

   width = df.x.max() - df.x.min() + 10
   height = df.y.max() - df.y.min() + 10

We group the dataframe by gene and create the list of gene names:

::

   grouped = df.groupby('target').agg(list)
   genes = list(grouped.index)

And finally the coordinate list:

::

   coord_list = []
   for target, coords in grouped.iterrows():
       coord_list.append(np.array(list(zip(*coords))))

Create the ``SSAMDataset`` object
---------------------------------

With everything in place we can now instantiate the ``SSAMDataset``
object:

::

   ds = ssam.SSAMDataset(genes, coord_list, width, height)

Now we can start the analysis with the `kernel density
estimation <kde.md>`__ step.
