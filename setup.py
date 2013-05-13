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
import textwrap
import ctypes
import sys
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
lib_dirs = []
inc_dirs = ['KMeansRMSD', np.get_include()]
default_header_dirs = []
default_library_dirs = None
default_runtime_dirs = None


##############################################################################
# Utilities
##############################################################################


def _print_admonition(kind, head, body):
    tw = textwrap.TextWrapper(initial_indent='   ', subsequent_indent='   ')

    print(".. %s:: %s" % (kind.upper(), head))
    for line in tw.wrap(body):
        print(line)


def print_warning(head, body=''):
    _print_admonition('warning', head, body)


def exit_with_error(head, body=''):
    _print_admonition('error', head, body)
    sys.exit(1)


def add_from_path(envname, dirs):
    try:
        dirs.extend(os.environ[envname].split(os.pathsep))
    except KeyError:
        pass


def add_from_flags(envname, flag_key, dirs):
    for flag in os.environ.get(envname, "").split():
        if flag.startswith(flag_key):
            dirs.append(flag[len(flag_key):])


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


def _find_file_path(name, locations, prefixes=[''], suffixes=['']):
    for prefix in prefixes:
        for suffix in suffixes:
            for location in locations:
                path = os.path.join(location, prefix + name + suffix)
                if os.path.isfile(path):
                    return path
    return None


class Package(object):
    def __init__(self, name, tag, header_name, library_name,
                 target_function=None):
        self.name = name
        self.tag = tag
        self.header_name = header_name
        self.library_name = library_name
        self.runtime_name = library_name
        self.target_function = target_function
        self.found = True

    def find_header_path(self, locations=default_header_dirs):
        return _find_file_path(self.header_name, locations, suffixes=['.h'])

    def find_library_path(self, locations=default_library_dirs):
        return _find_file_path(
            self.library_name, locations,
            self._library_prefixes, self._library_suffixes)

    def find_runtime_path(self, locations=default_runtime_dirs):
        """
        returns True if the runtime can be found
        returns None otherwise
        """
        # An explicit path can not be provided for runtime libraries.
        # (The argument is accepted for compatibility with previous methods.)

        # dlopen() won't tell us where the file is, just whether
        # success occurred, so this returns True instead of a filename
        for prefix in self._runtime_prefixes:
            for suffix in self._runtime_suffixes:
                try:
                    ctypes.CDLL(prefix + self.runtime_name + suffix)
                    return True
                except OSError:
                    pass

    def find_directories(self, location):
        dirdata = [
            (self.header_name, self.find_header_path, default_header_dirs),
            (self.library_name, self.find_library_path, default_library_dirs),
            (self.runtime_name, self.find_runtime_path, default_runtime_dirs),
        ]

        locations = []
        if location:
            # The path of a custom install of the package has been
            # provided, so the directories where the components
            # (headers, libraries, runtime) are going to be searched
            # are constructed by appending platform-dependent
            # component directories to the given path.
            # Remove leading and trailing '"' chars that can mislead
            # the finding routines on Windows machines
            locations = [os.path.join(location.strip('"'), compdir)
                                        for compdir in self._component_dirs]

        directories = [None, None, None]  # headers, libraries, runtime
        for idx, (name, find_path, default_dirs) in enumerate(dirdata):
            path = find_path(locations or default_dirs)
            if path:
                if path is True:
                    directories[idx] = True
                    continue

                # Take care of not returning a directory component
                # included in the name.  For instance, if name is
                # 'foo/bar' and path is '/path/foo/bar.h', do *not*
                # take '/path/foo', but just '/path'.  This also works
                # for name 'libfoo.so' and path '/path/libfoo.so'.
                # This has been modified to just work over include files.
                # For libraries, its names can be something like 'bzip2'
                # and if they are located in places like:
                #  \stuff\bzip2-1.0.3\lib\bzip2.lib
                # then, the directory will be returned as '\stuff' (!!)
                # F. Alted 2006-02-16
                if idx == 0:
                    directories[idx] = os.path.dirname(path[:path.rfind(name)])
                else:
                    directories[idx] = os.path.dirname(path)

        return tuple(directories)


class PosixPackage(Package):
    _library_prefixes = ['lib']
    _library_suffixes = ['.so', '.dylib', '.a']
    _runtime_prefixes = _library_prefixes
    _runtime_suffixes = ['.so', '.dylib']
    _component_dirs = ['include', 'lib']
_Package = PosixPackage


##############################################################################
# script
##############################################################################

# add some defaults
add_from_path("CPATH", default_header_dirs)
add_from_path("C_INCLUDE_PATH", default_header_dirs)
add_from_flags("CPPFLAGS", "-I", default_header_dirs)
default_header_dirs.extend(['/usr/include', '/usr/local/include'])

default_library_dirs = []
add_from_flags("LDFLAGS", "-L", default_library_dirs)
default_library_dirs.extend(
    os.path.join(_tree, _arch)
    for _tree in ('/usr/local', '/sw', '/opt', '/opt/local', '/usr', '/')
        for _arch in ('lib64', 'lib'))
default_runtime_dirs = default_library_dirs

if sys.platform.lower().startswith('darwin'):
    inc_dirs.extend(default_header_dirs)
    lib_dirs.extend(default_library_dirs)

gsl_package = _Package("GSL", 'GSL', 'gsl/gsl_matrix', 'gsl')
gsl_package.target_function = 'gsl_matrix_view_array'

gslcblas_package = _Package("GSL CBLAS", 'GSL', 'gsl/gsl_cblas', 'gslcblas')
gslcblas_package.target_function = 'cblas_dgemm'

# Allow setting the GSL dir and additional link flags either in
# the environment or on the command line.
# First check the environment...
GSL_DIR = os.environ.get('GSL_DIR', '')
CFLAGS = os.environ.get('CFLAGS', '').split()
LIBS = os.environ.get('LIBS', '').split()
LFLAGS = os.environ.get('LFLAGS', '').split()

# ...then the command line.
args = sys.argv[:]
for arg in args:
    if arg.find('--gsl=') == 0:
        GSL_DIR = os.path.expanduser(arg.split('=')[1])
        sys.argv.remove(arg)
    elif arg.find('--lflags=') == 0:
        LFLAGS = arg.split('=')[1].split()
        sys.argv.remove(arg)
    elif arg.find('--cflags=') == 0:
        CFLAGS = arg.split('=')[1].split()
        sys.argv.remove(arg)

##############################################################################
# find the packages GSL and GSL CBLAS libraries and headers
##############################################################################

compiler = new_compiler()
for package, location in [(gsl_package, GSL_DIR), (gslcblas_package, GSL_DIR)]:

    (hdrdir, libdir, rundir) = package.find_directories(location)

    if not libdir and package.target_function:
        libdir = compiler.has_function(package.target_function, libraries=(package.library_name,))

    if not (hdrdir and libdir):
        print_warning("* Could not find %s headers and library; " % package.name,
              "disabling support for it. You may need to explicitly state "
              "where your local %s headers and library can be found "
              "by setting the ``%s_DIR`` environment variable "
              "or by using the ``--%s`` command-line option." %
              (package.name, package.tag, package.tag.lower()))
        package.found = False
        continue

    if libdir in ("", True):
        print("* Found %s headers at ``%s``, the library is located in the "
              "standard system search dirs." % (package.name, hdrdir))
    else:
        print("* Found %s headers at ``%s``, library at ``%s``."
              % (package.name, hdrdir, libdir))

    if hdrdir not in default_header_dirs:
        inc_dirs.append(hdrdir)

    if libdir not in default_library_dirs and libdir not in ("", True):
        lib_dirs.append(libdir)

    if not rundir:
        loc = {
            'posix': "the default library paths.",
            'nt': "any of the directories in %%PATH%%.",
        }[os.name]

        print_warning(
            "Could not find the %s runtime." % package.name,
            "The %(name)s shared library was *not* found in %(loc)s "
            "In case of runtime problems, please remember to install it."
            % dict(name=package.name, loc=loc))

##############################################################################
# declare the extension modules
##############################################################################

extensions = []
if gsl_package.found:
    # only compile the cython extension code if they have the GSL
    extensions.append(Extension(
        "kmeansrmsd.clustering",
        sources=["KMeansRMSD/clustering.pyx", "KMeansRMSD/kmeans_rmsd_subroutines.c"],
        libraries=['m', 'gslcblas', 'gsl'],
        library_dirs=lib_dirs,
        extra_link_args=LFLAGS,
        extra_compile_args=CFLAGS,
        include_dirs = inc_dirs,
    ))

    extensions.append(Extension(
        "kmeansrmsd.kmeans_rmsd_subroutines_tests",
        sources=["KMeansRMSD/kmeans_rmsd_subroutines_tests.pyx",
                 "KMeansRMSD/kmeans_rmsd_subroutines.c"],
        libraries=['m', 'gslcblas', 'gsl'],
        library_dirs=lib_dirs,
        extra_link_args=LFLAGS,
        extra_compile_args=CFLAGS,
        include_dirs = inc_dirs,
    ))

##############################################################################
# run the setup script
##############################################################################

check_python_dependencies('mdtraj', 'msmbuilder', 'tables', 'yaml', 'sklearn', 'scipy')
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
