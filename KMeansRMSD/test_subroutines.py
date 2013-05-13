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

"""
Pure python wrapper of the tests declared in kmeans_rmsd_subroutines_tests.pyx,
the cython file. the nosetests testing framework doesn't normally detect tests
in cython files, so we wrap them up here in a test generator.
"""

###############################################################
# Imports
###############################################################

import sys
try:
    from kmeansrmsd import kmeans_rmsd_subroutines_tests
    HAVE_C_CODE = True
except:
    HAVE_C_CODE = False

###############################################################
# tests
###############################################################

def test_all_builtins():
    if not HAVE_C_CODE:
        return
        
    for name in dir(kmeans_rmsd_subroutines_tests):
        # grab every name from kmeans_rmsd_subroutines_tests that looks
        # like it might be a test function
        if name.startswith('test_'):
            t = getattr(kmeans_rmsd_subroutines_tests, name)
            # you cannot set an arbitrary attribute like "description"
            # on a function that's declared in c, so we put in in a little
            # pure python wrapper. this way the description of the test
            # shows up properly when you run nosetests in verbose mode
            f = _wrapper(t)
            f.description = name
                
            yield f

###############################################################
# utilities
###############################################################

def _wrapper(f):
    def inner():
        f()
    return inner
