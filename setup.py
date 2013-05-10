import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy.distutils.system_info import get_info

extra_link_args = []
extra_compile_args = []

blas_include = get_info('blas_opt')['extra_compile_args'][1][2:]
extra_compile_args.extend(get_info('blas_opt')['extra_compile_args'])
extra_link_args.extend(get_info('lapack_opt')['extra_link_args'])

kmeansrmsd = Extension("kmeansrmsd.clustering",
    sources=["main.pyx", "lib.cpp"],
    language="c++",
    libraries=['m', 'cblas', 'clapack'],
    extra_compile_args=list(set(extra_compile_args)),
    include_dirs = [".", np.get_include(), blas_include],
    )

kmeansrmsd_test = Extension("kmeansrmsd.test",
    sources=["test.pyx", "lib.cpp"],
    language="c++",
    libraries=['m', 'cblas', 'clapack'],
    extra_compile_args=list(set(extra_compile_args)),
    include_dirs = [".", np.get_include(), blas_include],
    )

setup(
    name='kmeansrmsd',
    packages={'kmeansrmsd': 'KMeansRMSD'},
    cmdclass = {'build_ext': build_ext},
    ext_modules =[kmeansrmsd, kmeansrmsd_test]
)