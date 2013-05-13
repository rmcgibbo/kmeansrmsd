# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

##############################################################################
# Imports
##############################################################################

import os
import sys
import tempfile
import shutil
from distutils.errors import LinkError, CompileError
from distutils.ccompiler import new_compiler
from distutils.extension import Extension
from Cython.Distutils import build_ext
from distutils.core import setup

import numpy as np
from numpy.distutils.system_info import get_info

###############################################################################
# Globals
##############################################################################

__version__ = '0.2'

##############################################################################
# Code
##############################################################################



def hasfunction(cc, funcname):
    """Check to see if the compiler/linker supports a certain function

    simplified version of distutils.ccompiler.CCompiler.has_function
    that actually removes its temporary files.
    http://stackoverflow.com/questions/7018879/disabling-output-when-compiling-with-distutils
    """
    tmpdir = tempfile.mkdtemp(prefix='rmsdkmeans-install-')
    devnull = oldstderr = oldstdout = None
    try:
        try:
            fname = os.path.join(tmpdir, 'funcname.c')
            f = open(fname, 'w')
            f.write('int main(void) {\n')
            f.write('    %s();\n' % funcname)
            f.write('}\n')
            f.close()
            # Redirect stderr to /dev/null to hide any error messages
            # from the compiler.
            # This will have to be changed if we ever have to check
            # for a function on Windows.
            devnull = open('/dev/null', 'w')
            devnull2 = open('/dev/null', 'w')
            oldstderr = os.dup(sys.stderr.fileno())
            oldstdout = os.dup(sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            os.dup2(devnull2.fileno(), sys.stdout.fileno())
            objects = cc.compile([fname], output_dir=tmpdir)
            cc.link_executable(objects, os.path.join(tmpdir, "a.out"))
            
        except (LinkError, CompileError):
            return False
        return True
    finally:
        if oldstderr is not None:
            os.dup2(oldstderr, sys.stderr.fileno())
        if oldstdout is not None:
            os.dup2(oldstdout, sys.stdout.fileno())
        if devnull is not None:
            devnull.close()
        shutil.rmtree(tmpdir)

def detect_gsl():
    compiler = new_compiler()
    compiler.add_library('gsl')
    if not hasfunction(compiler, 'gsl_matrix_view_array'):
        print '\033[91m#####################################################'
        print "ERROR: The GNU scientific library (libgsl) was not found"
        print "on your system. The parallel C implementation of this"
        print "code is therefore not available. If you've installed"
        print "the GSL in a nonstandard location, you may need to"
        print "set your %sLDLIBRARY_PATH env variable." % ('DY' if sys.platform == 'darwin' else '')
        print '#####################################################\033[0m'
        return False

    compiler.add_library('gslcblas')
    if not hasfunction(compiler, 'cblas_dgemm'):
        print '\033[91m#####################################################'
        print "ERROR: The GNU scientific library BLAS interface (libgslcblas)"
        print "was not found on your system. The parallel C implementation"
        print "of this code is therefore not available. If you've"
        print "installed the GSL in a nonstandard location, you may need to"
        print "set your %sLDLIBRARY_PATH env variable." % ('DY' if sys.platform == 'darwin' else '')
        print '#####################################################\033[0m'
        return False
        
    return True

def check_python_dependencies(*packages):
    for p in packages:
        try:
            __import__(p)
        except ImportError:
            print '\033[91m#####################################################'
            print "ERROR: This package requires the python package"
            print "'%s'" % p
            print '#####################################################\033[0m'
            sys.exit(1)


check_python_dependencies('mdtraj', 'msmbuilder', 'tables', 'yaml', 'sklearn', 'scipy')


extensions = []
if detect_gsl():
    # only compile the cython extension code if they have the GSL
    extensions.append(Extension(
        "kmeansrmsd.clustering",
        sources=["KMeansRMSD/clustering.pyx", "KMeansRMSD/kmeans_rmsd_subroutines.c"],
        libraries=['m', 'gslcblas', 'gsl'],
        extra_compile_args=['-Wall'],
        include_dirs = ["KMeansRMSD", np.get_include()],
    ))

    extensions.append(Extension(
        "kmeansrmsd.kmeans_rmsd_subroutines_tests",
        sources=["KMeansRMSD/kmeans_rmsd_subroutines_tests.pyx",
                 "KMeansRMSD/kmeans_rmsd_subroutines.c"],
        extra_compile_args=['-Wall'],
        libraries=['m', 'gslcblas', 'gsl'],
        include_dirs = ["KMeansRMSD", np.get_include()],
    ))

setup(
    name='kmeansrmsd',
    version=__version__,
    packages=['kmeansrmsd',],
    package_dir={'kmeansrmsd':'KMeansRMSD'},
    cmdclass = {'build_ext': build_ext},
    ext_modules = extensions,

    description='Protein conformational clustering with RMSD by K-Means',
    author='Robert McGibbon',
    author_email='rmcgibbo@gmail.com',
    license='GPL3',
    classifiers=['Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
        'Operating System :: POSIX'
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Software Development :: Libraries :: Python Modules'],
)
