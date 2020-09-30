import io
import setuptools
try:
    import numpy as np
except ImportError:
    print("Please install Numpy first. e.g. pip install numpy")
    exit(1)
try:
    import torch
except ImportError:
    print("Please install Pytorch first. Instructions: https://pytorch.org/get-started/locally/")
    exit(1)
from glob import glob

module_utils = setuptools.extension.Extension('ssam.utils', sources=["c/utils.cpp"], extra_compile_args=["-fopenmp"], extra_link_args=["-fopenmp"], include_dirs=[np.get_include()])

with io.open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ssam",
    version="1.1.0",
    author="Jeongbin Park",
    author_email="j.park@dkfz-heidelberg.de",
    description="SSAM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eilslabs/ssam",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Operating System :: POSIX",
    ],
    ext_modules = [module_utils],
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "umap-learn",
        "python-louvain",
        "sparse",
        "scikit-image",
        "pyarrow",
        "pyyaml",
        "dask",
        "zarr",
        "zarr[array]",
        "hdbscan",
        "torch",
    ]
)
