cimport numpy as np
from cython.parallel import prange

import sys
import time
import itertools
import numpy as np
import scipy.linalg

from mdtraj.geometry.alignment import rmsd_kabsch
from mdtraj import IRMSD
from mdtraj.geometry import rmsd

cdef extern from "lib.h":
    int gower_matrix(double* X, int X_dim0, int X_dim1, int X_dim2,
        long* assignments, int assignments_dim0, long k,
        double* B, int B_dim0, int B_dim1)  nogil
    int average_structure(double* X, int X_dim0, int X_dim1, int X_dim2,
                      long* assignments, int assignments_dim0, long k,
                      double* R, int R_dim0, int R_dim1) nogil
    int is_mirror_image(double* X, int X_dim0, int X_dim1, double* Y, int Y_dim0, int Y_dim1) nogil
    int center_inplace(double* X, int X_dim0, int X_dim1) nogil
    int rectify_mirror(double* X, int X_dim0, int X_dim1, double* Y, int Y_dim0, int Y_dim1) nogil


##############################################################################
# Pure python implementations (reference) of the code in lib.cpp
##############################################################################

def _gower_matrix(X):
    B = sum(np.dot(x, x.T) for x in X) / float(len(X))   
    b = B.mean(1)
    bb = b.mean()
    return (B - np.add.outer(b, b)) + bb

def _is_mirror_image(X, Y):
    ## SVD of correlation matrix
    
    # print '(Python) covariance', np.dot(X.T, Y)
    V, L, U = scipy.linalg.svd(np.dot(X.T, Y))
    R = np.dot(V, U)
    
    # print '(Python) Determinant', scipy.linalg.det(R)
    if scipy.linalg.det(R) < 0:
        return 1
    return 0

def _average_structure(X, reference=None):
    B = _gower_matrix(X)
    v, U = scipy.linalg.eigh(B)
    if np.iscomplex(v).any():
        v = v.real
    if np.iscomplex(U).any():
        U = U.real

    indices = np.argsort(v)[-3:]
    v = np.take(v, indices, 0)
    U = np.take(U, indices, 1)
    x = U * np.sqrt(v)    

    x -= x.mean(0)
    
    if reference is None:
        reference = X[-1]
    reference -= reference.mean()
    
    x = _rectify_mirror(x, reference)
    if _is_mirror_image(x, reference):
        raise ValueError('It\'s a trap!')
    
    return x
    
def _rectify_mirror(X, Y):
    i = 0
    while _is_mirror_image(X, Y) and i < 2:
        X[:, i] *= -1
        i += 1
    return X

##############################################################################
# Tests for each of the functions in lib.cpp, comparing it with the pure
# python
##############################################################################

def test_gower_matrix():
    n_frames = 10
    n_atoms = 5

    cdef double[:, :, :] X = np.random.randn(n_frames, n_atoms, 3)
    cdef long[:] assignments = np.zeros(n_frames, dtype=np.int64)
    cdef long k = 0

    _B = _gower_matrix(np.asarray(X))

    cdef double[:, :] B = np.zeros((n_atoms, n_atoms), dtype=np.double)
    
    gower_matrix(&X[0,0,0], X.shape[0], X.shape[1], X.shape[2],
                 &assignments[0], assignments.shape[0], k,
                 &B[0,0], B.shape[0], B.shape[1])
                 
    np.testing.assert_array_almost_equal(np.asarray(B), _B)
    print 'test_gower_matrix passed'

def test_center_inplace():
     cdef double[:, :] X = np.random.randn(100, 3)
     X2 = np.copy(np.asarray(X))
     
     center_inplace(&X[0,0], X.shape[0], X.shape[1])

     X2 -= X2.mean(0)
     
     np.testing.assert_array_almost_equal(np.asarray(X), X2)
     print 'test_center_inplace passed'

def test_is_mirror_image():
    cdef double[:, :] X
    cdef double[:, :] Y
    
    for i in range(5):
        X = np.random.randn(100, 3)
        Y = np.random.randn(100, 3)

        X = X - np.mean(X, 0)
        Y = Y - np.mean(Y, 0)
    
        result1 = is_mirror_image(&X[0,0], X.shape[0], X.shape[1],
                                  &Y[0,0], Y.shape[0], Y.shape[1])
        result2 = _is_mirror_image(X, Y)
        assert result1 == result2
    print 'test_is_mirror_image passed'


def test_average_structure():
    
    n_frames = 10
    n_atoms = 5

    cdef double[:, :, :] X = np.random.randn(n_frames, n_atoms, 3)
    X2 = np.asarray(X).copy()
    cdef long[:] assignments = np.zeros(n_frames, dtype=np.int64)
    cdef long k = 0
    cdef double[:, :] result = np.zeros((n_atoms, 3), dtype=np.double)

    average_structure(&X[0,0,0], X.shape[0], X.shape[1], X.shape[2],
                 &assignments[0], assignments.shape[0], k,
                 &result[0,0], result.shape[0], result.shape[1])


    r2 = _average_structure(X2)
    difference = rmsd_kabsch(np.asarray(result), r2)
    assert difference < 1e-10

    print 'test_average_structure passed'

def test_mirror_flipper():
    cdef np.ndarray[double, ndim=2] conf1 = np.array([[  2.65586755e-01,  -9.27238597e-01,  -1.88187143e+00],
       [  5.86940362e-01,  -1.01088117e+00,  -8.27745500e-04],
       [ -4.84376667e-01,  -7.68769912e-01,   2.55097302e+00],
       [ -8.75585443e-01,   6.06322965e-01,   3.11763150e-01],
       [  1.49247711e-01,   1.64926174e-01,   4.32468497e-01],
       [ -1.53470843e-01,  -4.55160757e-01,  -3.85001867e-01],
       [  1.05591533e+00,   9.48715369e-01,   1.14833950e+00],
       [ -3.71162052e-01,  -2.00374187e-01,  -6.39289609e-01],
       [ -2.48058346e-01,   9.15185423e-01,  -1.70458350e+00],
       [  7.49631938e-02,   7.27274697e-01,   1.68029988e-01]])

    cdef np.ndarray[double, ndim=2] conf2 = np.array([[ -2.65586755e-01,  -9.27238597e-01,  -1.88187143e+00],
           [ -5.86940362e-01,  -1.01088117e+00,  -8.27745500e-04],
           [  4.84376667e-01,  -7.68769912e-01,   2.55097302e+00],
           [  8.75585443e-01,   6.06322965e-01,   3.11763150e-01],
           [ -1.49247711e-01,   1.64926174e-01,   4.32468497e-01],
           [  1.53470843e-01,  -4.55160757e-01,  -3.85001867e-01],
           [ -1.05591533e+00,   9.48715369e-01,   1.14833950e+00],
           [  3.71162052e-01,  -2.00374187e-01,  -6.39289609e-01],
           [  2.48058346e-01,   9.15185423e-01,  -1.70458350e+00],
           [ -7.49631938e-02,   7.27274697e-01,   1.68029988e-01]])

    cdef np.ndarray[double, ndim=2] reference = np.array([[ 0.82221491, -0.36405214,  1.05510652],
           [-0.11809122,  0.40392022,  1.44847452],
           [-0.55748556, -1.19532741, -0.03597826],
           [-0.24324392, -0.32296946,  1.48038213],
           [-0.13996595,  1.17421015, -0.39945472],
           [ 0.58941451, -0.42541861, -1.39731127],
           [ 2.63017262, -0.04900264, -0.11425032],
           [-0.22778791, -0.30019201, -0.6070062 ],
           [-1.20558585,  0.90675396, -0.8028707 ],
           [-1.54964163,  0.17207795, -0.62709169]])
    
    print 'centroid1', np.mean(conf1, 0)
    print 'centroid2', np.mean(conf2, 0)
    print 'centroid reference', np.mean(reference, 0)
    
    print '_is_mirror_image(conf1, reference)', _is_mirror_image(conf1, reference)
    print '_is_mirror_image(conf2, reference)', _is_mirror_image(conf2, reference)
    
    print 'is_mirror_image(conf1, reference)',
    print is_mirror_image(&conf1[0,0], conf1.shape[0], conf1.shape[1],
                          &reference[0,0], reference.shape[0], reference.shape[1])

    print 'is_mirror_image(conf2, reference)',
    print is_mirror_image(&conf2[0,0], conf2.shape[0], conf2.shape[1],
                          &reference[0,0], reference.shape[0], reference.shape[1])
    
    print 'conf1'
    print conf1
    cdef np.ndarray[double, ndim=2] conf1_copy = np.copy(conf1)
    _rectify_mirror(conf1_copy, reference)
    
    print 'rectified conf1 (pyhon)'
    print conf1_copy
    
    cdef np.ndarray[double, ndim=2] conf1_copy2 = np.copy(conf1)
    rectify_mirror(&conf1_copy2[0,0], conf1_copy2.shape[0], conf1_copy2.shape[1],
                   &reference[0,0], reference.shape[0], reference.shape[1])
    
    print 'rectified conf1 (C)'
    print conf1_copy2
    
    np.testing.assert_array_almost_equal(conf1_copy2, conf1_copy)
    print 'passed'
    

def test():
    test_gower_matrix()
    test_center_inplace()
    test_is_mirror_image()
    test_average_structure()
    test_mirror_flipper()

##############################################################################
# Main code
##############################################################################


def kmeans_mds(np.ndarray[double, ndim=3] xyzlist, int k, n_max_iters=100, threshold=1e-8):
    """k-means clustering with the RMSD distance metric.

    this is an iterative algorithm. during each iteration we first move each cluster center to
    the empirical average of the conformations currently assigned to it, and then we re-assign
    all of the conformations given the new locations of the centers.
    
    to compute the average conformations, we use a form of classical multidimensional
    scaling / principle coordinate analysis.
    """
    assert xyzlist.shape[2] == 3

    # setup for the rmsd calculation
    n_frames, n_atoms = xyzlist.shape[0], xyzlist.shape[1]
    xyzlist_G = rmsd.calculate_G(np.asarray(xyzlist))
    xyzlist_irmsd, n_atoms_padded = rmsd.reshape_irmsd(np.asarray(xyzlist))
    
    # setup for the clustering stuff
    # start with just some random assignments (most stuff unassigned), each
    # cluster only a single state
    cdef np.ndarray[long, ndim=1] assignments = -1*np.ones(n_frames, dtype=np.int64)
    assignments[0:k] = np.arange(k)
    np.random.shuffle(assignments)
    
    # the j-th cluster has cartesian coorinates centers[j]
    cdef np.ndarray[double, ndim=3] centers = np.zeros((k, n_atoms, 3))
    # assignment_dist[i] gives the RMSD between the ith conformation and its cluster center
    assignment_dist = np.inf * np.ones(n_frames, dtype=np.float32)
    scores = [np.inf]
    times = [time.time()]
    

    for n in itertools.count():
        # recenter each cluster based on its current members
        for i in prange(k, nogil=True):
            average_structure(&xyzlist[0,0,0], xyzlist.shape[0], xyzlist.shape[1], xyzlist.shape[2],
                &assignments[0], assignments.shape[0], i,
                &centers[i, 0, 0], centers.shape[1], centers.shape[2])

        # prepare the new centers for RMSD
        centers_G = rmsd.calculate_G(np.asarray(centers))
        centers_irmsd, _ = rmsd.reshape_irmsd(np.asarray(centers))
        
        # reassign all of the data
        assignments = -1 * np.ones(n_frames, dtype=np.int64)
        assignment_dist = np.inf * np.ones(n_frames, dtype=np.float32)
        for i in range(k):
            d = IRMSD.rmsd_one_to_all(centers_irmsd, xyzlist_irmsd, centers_G, xyzlist_G, n_atoms, i)
            where = d < assignment_dist
            assignments[where] = i
            assignment_dist[where] = d[where]

        # check how far each cluster center moved during the lrm iteration
        # and break if necessary
        scores.append(np.sqrt(np.mean(np.square(assignment_dist))))
        times.append(time.time())
        print 'round %3d, RMS(data<->center) %8f, change %.3e' % (n, scores[n], scores[-1] - scores[-2])
        if threshold is not None and scores[-2] - scores[-1] < threshold:
            print 'score decreased less than threshold (%s). done' % threshold
            break
        if n_max_iters is not None and n >= n_max_iters:
            print 'reached maximum number of iterations. done'
            break
    
    print 'Performance Summary'
    print 'mean time per round (s)   %.4f' % np.mean(np.diff(times))
    print 'stddev time per round (s) %.4f' % np.std(np.diff(times))
    print 'total time (s)            %.4f' % (times[-1] - times[0])
    return centers, assignments, assignment_dist, np.array(scores), np.array(times)


def test_kmeans():
    import pyrr
    dataset = []
    for i in range(10):
        xyz = np.random.randn(10,3)
        for j in range(10):
            rot = pyrr.matrix33.create_from_quaternion(pyrr.quaternion.create_from_eulers(np.random.rand(3)*2*np.pi))
            dataset.append(np.dot(xyz + 0.01*np.random.randn(*xyz.shape), rot))
    dataset = np.array(dataset)

    centers, assignments, distances, scores, times = kmeans_mds(dataset, k=10)
    print assignments

#test()
test_kmeans()