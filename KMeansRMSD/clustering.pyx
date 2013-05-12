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
# Python/Cython imports
##############################################################################

cimport numpy as np
from cython.parallel import prange

import sys
import time
import itertools
import numpy as np

from kmeansrmsd.test import _average_structure
from mdtraj import IRMSD
from mdtraj.geometry import rmsd

__all__ = ['kmeans_mds']

##############################################################################
# C headers
##############################################################################

cdef extern from "kmeans_rmsd_subroutines.h":
    int average_structure(double* X, int X_dim0, int X_dim1, int X_dim2, int X_dim2_mem,
                          long* assignments, int assignments_dim0, long k,
                          double* R, int R_dim0, int R_dim1, int R_dim1_mem) nogil

##############################################################################
# main clustering algorithm.
##############################################################################

def kmeans_mds(np.ndarray[double, ndim=3] xyzlist, int n_real_atoms, int k, max_iters=10, max_time=None, threshold=1e-8):
    """k-means clustering with the RMSD distance metric.

    this is an iterative algorithm. during each iteration we first move each cluster center to
    the empirical average of the conformations currently assigned to it, and then we re-assign
    all of the conformations given the new locations of the centers.

    to compute the average conformations, we use a form of classical multidimensional
    scaling / principle coordinate analysis.
    """
    assert xyzlist.shape[1] == 3, 'xyzlist must be n_frames, 3, n_atoms'
    assert xyzlist.shape[2] % 4 == 0, 'number of atoms must be a multiple of four. you can pad with zeros to get up to that.'

    # static type declarations
    cdef int n_frames, n_padded_atoms, n, i
    cdef np.ndarray[np.float32_t, ndim=1] xyzlist_g             # for RMSD
    cdef np.ndarray[np.float32_t, ndim=3] xyzlist_float         # single precsion copy

    cdef np.ndarray[long, ndim=1] assignments
    cdef np.ndarray[double, ndim=3] centers
    cdef np.ndarray[np.float32_t, ndim=1] centers_g             # for RMSD
    cdef np.ndarray[np.float32_t, ndim=3] centers_float         # single precision copy for RMSD
    cdef np.ndarray[np.float32_t, ndim=1] assignment_dist
    cdef np.ndarray[np.float32_t, ndim=1] d
    float32_max = np.finfo(np.float32).max
    
    n_frames, n_padded_atoms = xyzlist.shape[0], xyzlist.shape[2]
    remove_center_of_mass(xyzlist, n_real_atoms)
    xyzlist_g = calculate_g(xyzlist)
    assert not np.any(np.logical_or(np.isinf(xyzlist_g), np.isnan(xyzlist_g))), 'nan in xyzlist G'
    xyzlist_float = np.asarray(xyzlist, order='C', dtype=np.float32)

    # start with just some random assignments (most stuff unassigned), each
    # cluster only a single state
    assignments = -1*np.ones(n_frames, dtype=np.int64)
    assignments[0:k] = np.arange(k)
    np.random.shuffle(assignments)

    centers = np.zeros((k, 3, n_padded_atoms), dtype=np.float64)
    centers_float = np.zeros((k, 3, n_padded_atoms), dtype=np.float32)
    assignment_dist = float32_max * np.ones(n_frames, dtype=np.float32)

    scores = [np.inf]
    times = [time.time()]
    
    for n in itertools.count():
        # recenter each cluster based on its current members
        for i in prange(k, nogil=True):
            average_structure(&xyzlist[0,0,0], xyzlist.shape[0], xyzlist.shape[1], n_real_atoms, n_padded_atoms,
               &assignments[0], assignments.shape[0], i,
               &centers[i, 0, 0], centers.shape[1], n_real_atoms, n_padded_atoms)

        # prepare the new centers for RMSD
        centers_g = calculate_g(centers)
        assert not np.any(np.logical_or(np.isinf(centers_g), np.isnan(centers_g))), 'nan in centers G'
        centers_float = np.asarray(centers, dtype=np.float32)

        # reassign all of the data
        assignment_dist[:] = float32_max
        for i in range(k):
            d = IRMSD.rmsd_one_to_all(centers_float, xyzlist_float, centers_g, xyzlist_g, n_real_atoms, i)
            where = d < assignment_dist
            assignments[where] = i
            assignment_dist[where] = d[where]

        # check how far each cluster center moved during the lrm iteration
        # and break if necessary
        scores.append(np.sqrt(np.mean(np.square(assignment_dist))))
        times.append(time.time())
        print 'round %3d, RMS radius %8f, change %.3e' % (n, scores[-1], scores[-1] - scores[-2])
        if threshold is not None and scores[-2] - scores[-1] < threshold:
            print 'score decreased less than threshold (%s). done\n' % threshold
            break
        if max_iters is not None and n >= max_iters:
            print 'reached maximum number of iterations. done\n'
            break
        if max_time is not None and times[-1] >= times[0] + max_time:
            print 'reached maximum amount of time. done\n'
            break

    print 'RMSD KMeans Summary (C)'
    print '-----------------------'
    print 'n frames:                  %d' % n_frames
    print 'n atoms(padded):           %d' % n_padded_atoms
    print 'n atoms (real):            %d' % n_real_atoms
    print 'n states:                  %d' % k
    print 'n rounds:                  %d' % (n + 1)
    print 'mean time per round (s):   %.4f' % np.mean(np.diff(times))
    print 'stddev time per round (s): %.4f' % np.std(np.diff(times))
    print 'total time (s):            %.4f' % (times[-1] - times[0])

    return centers, assignments, assignment_dist, np.array(scores), np.array(times)


##############################################################################
# C level utility functions
##############################################################################


cdef np.ndarray[np.float32_t, ndim=1] calculate_g(np.ndarray[double, ndim=3] xyzlist):
    """Calculate the trace of each frame's inner product matrix, for the RMSD
    calculation
    
    Parameters
    ----------
    xyzlist : np.ndarray, shape=(n_frames, 3, n_atoms)
        The cartesian coordinate. They should be already centered.
    
    Returns
    -------
    g : np.ndarray, shape=(n_frames), dtype=np.float32
        The inner product list, for each frame
    """
    assert xyzlist.shape[1] == 3, 'second dimension must be 3'
    assert xyzlist.dtype == np.float64, 'this should be double precision'
    
    cdef int i, j, k
    cdef double g_i
    cdef int n_frames = xyzlist.shape[0]
    cdef int n_spatial = xyzlist.shape[1]
    cdef int n_atoms = xyzlist.shape[2]

    cdef np.ndarray[np.float32_t, ndim=1] g = np.zeros(n_frames, dtype=np.float32)
    
    for i in range(n_frames):
        g_i = 0
        for j in range(n_spatial):
            for k in range(n_atoms):
                g_i += xyzlist[i, j, k]*xyzlist[i, j, k]
        g[i] = g_i
                
    return g

cdef remove_center_of_mass(np.ndarray[double, ndim=3] xyzlist, int n_real_atoms):
    """Remove the center of mass from a set of frames declared in atom major
    ordering (with padding atoms). Acts inplace.
    
    Parameters
    ----------
    xyzlist : np.ndarray, shape=(n_frames, 3, n_atoms_with_padding)
        The cartesian coordinates of each frame, in atom major order. Because RMSD
        requires that the number of atoms be a multiple of four, the 3rd dimension
        can potentially be longer than the actual number of real atoms.
    n_real_atoms : int
        The number of actual atoms in the sytem.
    """
    assert xyzlist.shape[1] == 3, 'second dimension must be 3'
    cdef int i, j
    cdef double muX, muY, muZ
    cdef int n_frames = xyzlist.shape[0]
    print 'removing center of mass...'
    for i in range(n_frames):
        muX = 0
        muY = 0
        muZ = 0
        for j in range(n_real_atoms):
            muX += xyzlist[i, 0, j]
            muY += xyzlist[i, 1, j]
            muZ += xyzlist[i, 2, j]
        muX /= n_real_atoms
        muY /= n_real_atoms
        muZ /= n_real_atoms
        
        for j in range(n_real_atoms):
            xyzlist[i, 0, j] -= muX
            xyzlist[i, 1, j] -= muY
            xyzlist[i, 2, j] -= muZ
