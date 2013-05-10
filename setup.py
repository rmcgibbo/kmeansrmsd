import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy.distutils.system_info import get_info

kmeansrmsd = Extension("kmeansrmsd.clustering",
    sources=["main.pyx", "lib.c"],
    libraries=['m', 'gslcblas', 'gsl'],
    include_dirs = [".", np.get_include()],
    )

kmeansrmsd_test = Extension("kmeansrmsd.test",
    sources=["test.pyx", "lib.c"],
    libraries=['m', 'gslcblas', 'gsl'],
    include_dirs = [".", np.get_include()],
    )

setup(
    name='kmeansrmsd',
    packages={'kmeansrmsd': 'KMeansRMSD'},
    cmdclass = {'build_ext': build_ext},
    ext_modules =[kmeansrmsd, kmeansrmsd_test]
)