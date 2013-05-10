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

##############################################################################
# C++ headers
##############################################################################

cdef extern from "kmeans_rmsd_subroutines.h":
    int average_structure(double* X, int X_dim0, int X_dim1, int X_dim2,
                      long* assignments, int assignments_dim0, long k,
                      double* R, int R_dim0, int R_dim1) nogil

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
    
    cdef int i
    cdef int n_frames = xyzlist.shape[0]
    cdef np.ndarray[np.float32_t, ndim=1] g = np.empty(n_frames, dtype=np.float32)
    
    for i in range(n_frames):
        g[i] = (xyzlist[i]**2).sum()

    return g

cdef remove_center_of_mass(np.ndarray[double, ndim=3] xyzlist):
    assert xyzlist.shape[1] == 3, 'second dimension must be 3'
    cdef int i
    cdef int n_frames = xyzlist.shape[0]
    for i in range(n_frames):
        muX, muY, muZ = np.mean(xyzlist[i], axis=1)
        xyzlist[i, 0, :] -= muX
        xyzlist[i, 1, :] -= muY
        xyzlist[i, 2, :] -= muZ


##############################################################################
# Main functions
##############################################################################

def kmeans_mds(np.ndarray[double, ndim=3] xyzlist, int k, n_max_iters=100, threshold=1e-8):
    """k-means clustering with the RMSD distance metric.

    this is an iterative algorithm. during each iteration we first move each cluster center to
    the empirical average of the conformations currently assigned to it, and then we re-assign
    all of the conformations given the new locations of the centers.

    to compute the average conformations, we use a form of classical multidimensional
    scaling / principle coordinate analysis.
    """
    assert xyzlist.shape[1] == 3, 'xyzlist must be n_frames, 3, n_atoms'

    # static type declarations
    cdef int n_frames, n_atoms, n, i
    cdef np.ndarray[np.float32_t, ndim=1] xyzlist_g             # for RMSD
    cdef np.ndarray[np.float32_t, ndim=3] xyzlist_float         # single precsion copy

    cdef np.ndarray[long, ndim=1] assignments
    cdef np.ndarray[double, ndim=3] centers
    cdef np.ndarray[np.float32_t, ndim=1] centers_g             # for RMSD
    cdef np.ndarray[np.float32_t, ndim=3] centers_float         # single precision copy for RMSD
    cdef np.ndarray[np.float32_t, ndim=1] assignment_dist
    cdef np.ndarray[np.float32_t, ndim=1] d
    float32_max = np.finfo(np.float32).max
    
    
    n_frames, n_atoms = xyzlist.shape[0], xyzlist.shape[2]
    xyzlist_g = calculate_g(xyzlist)
    remove_center_of_mass(xyzlist)
    xyzlist_float = np.asarray(xyzlist, order='C', dtype=np.float32)

    # start with just some random assignments (most stuff unassigned), each
    # cluster only a single state
    assignments = -1*np.ones(n_frames, dtype=np.int64)
    assignments[0:k] = np.arange(k)
    np.random.shuffle(assignments)

    centers = np.empty((k, 3, n_atoms), dtype=np.float64)
    centers_float = np.empty((k, 3, n_atoms), dtype=np.float32)
    assignment_dist = float32_max * np.ones(n_frames, dtype=np.float32)

    scores = [np.inf]
    times = [time.time()]

    for n in itertools.count():
        # recenter each cluster based on its current members
        for i in prange(k, nogil=True):
            average_structure(&xyzlist[0,0,0], xyzlist.shape[0], xyzlist.shape[1], xyzlist.shape[2],
                &assignments[0], assignments.shape[0], i,
                &centers[i, 0, 0], centers.shape[1], centers.shape[2])

        # prepare the new centers for RMSD
        centers_g = calculate_g(centers)
        centers_float = np.asarray(centers, dtype=np.float32)

        # reassign all of the data
        assignment_dist = float32_max * np.ones(n_frames, dtype=np.float32)
        for i in range(k):
            d = IRMSD.rmsd_one_to_all(centers_float, xyzlist_float, centers_g, xyzlist_g, n_atoms, i)
            where = d < assignment_dist
            assignments[where] = i
            assignment_dist[where] = d[where]

        # check how far each cluster center moved during the lrm iteration
        # and break if necessary
        scores.append(np.sqrt(np.mean(np.square(assignment_dist))))
        times.append(time.time())
        print 'round %3d, RMS radius %8f, change %.3e' % (n, scores[n], scores[-1] - scores[-2])
        if threshold is not None and scores[-2] - scores[-1] < threshold:
            print 'score decreased less than threshold (%s). done' % threshold
            break
        if n_max_iters is not None and n >= n_max_iters:
            print 'reached maximum number of iterations. done'
            break

    print '\nRMSD KMeans Performance Summary'
    print '--------------------------------'
    print 'mean time per round (s)   %.4f' % np.mean(np.diff(times))
    print 'stddev time per round (s) %.4f' % np.std(np.diff(times))
    print 'total time (s)            %.4f' % (times[-1] - times[0])

    return centers, assignments, assignment_dist, np.array(scores), np.array(times)

