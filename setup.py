import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy.distutils.system_info import get_info
extensions = []

extensions.append(Extension(
    "kmeansrmsd.clustering",
    sources=["KMeansRMSD/clustering.pyx", "KMeansRMSD/kmeans_rmsd_subroutines.c"],
    libraries=['m', 'gslcblas', 'gsl'],
    extra_compile_args=['-Wall'],
    include_dirs = [".", np.get_include()],
))

extensions.append(Extension(
    "kmeansrmsd.kmeans_rmsd_subroutines_tests",
    sources=["KMeansRMSD/kmeans_rmsd_subroutines_tests.pyx",
             "KMeansRMSD/kmeans_rmsd_subroutines.c"],
    extra_compile_args=['-Wall'],
    libraries=['m', 'gslcblas', 'gsl'],
    include_dirs = [".", np.get_include()],
))

setup(
    name='kmeansrmsd',
    packages={'kmeansrmsd': 'KMeansRMSD'},
    cmdclass = {'build_ext': build_ext},
    ext_modules = extensions
)