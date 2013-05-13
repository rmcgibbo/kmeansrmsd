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
    """k-means clustering with RMSD

    This algorithm uses a multidimensional scaling approach to "average" conformations
    for the kmeans update step. This iterative algorithm will finish when the
    first of the three possible convergence criteria is met.

    Parameters
    ----------
    xyzlist : np.ndarray, dtype=double, shape=(n_frames, 3, n_padded_atoms)
        The cartesian coordinates, in double precision, laid out in axis-major
        order. The number of atoms should be a multiple of four, with "padding"
        atoms inserted at (0,0,0) to get up to the next multiple of four if you
        have fewer.
    n_real_atoms : int
        The actual number of atoms you have, (i.e. without the padding atoms)
    k : int
        The number of clusters you'd like
    max_iters : int
        Convergence criteria. quit when this many iterations have been run.
    max_time : float
        Convergence criteria. quit when the elapsed time has exceeds this amount,
        in seconds.
    threshold : float
        convergence crieria. quit when the change in the RMS cluster radius in
        a round is less than this value.

    Returns
    -------
    centers : np.ndarray, shape=(n_frames, 3, n_padded_atoms)
        The cartesian coordinates of the cluster centers. Obviously these will
        only include the atoms that were used as input in xyzlist. Padding
        atoms will still be present.
    assignments : np.ndarray, shape=(n_frames,) dtype=long
        contains a 1D mapping of which conformation from xyzlist are
        assigned to which center.
    distances : np.ndarray, shape=(n_frames,) dtype=float
        contains the distance from each data point to its cluster center
    scores : np.ndarray, shape=(n_rounds,), dtyoe=float
        The RMS cluster radius at each iteration of the algorithm. Gives
        a sense of how quickly the algorithm was converging.
    times : np.ndarray, shape=(n_rounds,), dtype=float give information
        The wall clock time (seconds since unix epoch) when each round of the
        clustering algorithm was completed, so you can check the convergence
        vs. elapsed wall clock time.
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
    dremove_center_of_mass(xyzlist, n_real_atoms)
    xyzlist_g = dcalculate_g(xyzlist)
    assert not np.any(np.logical_or(np.isinf(xyzlist_g), np.isnan(xyzlist_g))), 'nan in xyzlist G'
    xyzlist_float = np.asarray(xyzlist, order='C', dtype=np.float32)

    # start with just some random assignments (most stuff unassigned), each
    # cluster only a single state
    assignments = np.empty(n_frames, dtype=np.int64)
    assignments.fill(-1)
    assignments[0:k] = np.arange(k)
    np.random.shuffle(assignments)

    centers = np.zeros((k, 3, n_padded_atoms), dtype=np.float64)
    centers_float = np.zeros((k, 3, n_padded_atoms), dtype=np.float32)
    assignment_dist = np.empty(n_frames, dtype=np.float32)
    assignment_dist.fill(np.inf)

    scores = [np.inf]
    times = [time.time()]
    
    for n in itertools.count():
        # recenter each cluster based on its current members
        for i in prange(k, nogil=True):
            average_structure(&xyzlist[0,0,0], xyzlist.shape[0], xyzlist.shape[1], n_real_atoms, n_padded_atoms,
               &assignments[0], assignments.shape[0], i,
               &centers[i, 0, 0], centers.shape[1], n_real_atoms, n_padded_atoms)

        # prepare the new centers for RMSD
        centers_g = dcalculate_g(centers)
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


cpdef np.ndarray[np.float32_t, ndim=1] dcalculate_g(np.ndarray[double, ndim=3] xyzlist):
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


cpdef np.ndarray[np.float32_t, ndim=1] scalculate_g(np.ndarray[np.float32_t, ndim=3] xyzlist):
    """Calculate the trace of each frame's inner product matrix, for the RMSD
    calculation. This is in single.
    
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


cpdef dremove_center_of_mass(np.ndarray[double, ndim=3] xyzlist, int n_real_atoms):
    """Remove the center of mass from a set of frames declared in atom major
    ordering (with padding atoms). Acts inplace. This is for double precision
    
    Parameters
    ----------
    xyzlist : np.ndarray, dtype=np.float64_t, shape=(n_frames, 3, n_atoms_with_padding)
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

cpdef sremove_center_of_mass(np.ndarray[np.float32_t, ndim=3] xyzlist, int n_real_atoms):
    """Remove the center of mass from a set of frames declared in atom major
    ordering (with padding atoms). Acts inplace. This is for single precision.
    The accumulating is done in double though.
    
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
        muX = 0.0
        muY = 0.0
        muZ = 0.0
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
