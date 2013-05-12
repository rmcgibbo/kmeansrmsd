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
Tests for all of the functions in keans_rmsd_subroutines. Because this
is a cython extension module, this code isn't automatically "found" by
nosetests, so it's wrapped by test_subroutines.py, which will yield
any test from this file, using the standard nosetests naming conventions,
to nose.
"""

cimport numpy as np

import sys
import time
import itertools
import numpy as np
import scipy.linalg

from mdtraj.geometry.alignment import rmsd_kabsch


cdef extern from "kmeans_rmsd_subroutines.h":
    int gower_matrix(double* X, int X_dim0, int X_dim1, int X_dim2, int X_dim2_mem,
        long* assignments, int assignments_dim0, long k,
        double* B, int B_dim0, int B_dim1)
    int average_structure(double* X, int X_dim0, int X_dim1, int X_dim2, int X_dim2_mem,
                          long* assignments, int assignments_dim0, long k,
                          double* R, int R_dim0, int R_dim1, int R_dim1_mem)
    int is_mirror_image(double* X, int X_dim0, int X_dim1, int X_dim1_mem,
                        double* Y, int Y_dim0, int Y_dim1, int Y_dim1_mem)
    int rectify_mirror(double* X, int X_dim0, int X_dim1, int X_dim1_mem,
                       double* Y, int Y_dim0, int Y_dim1, int Y_dim1_mem)
    int center_inplace(double* X, int X_dim0, int X_dim1, int X_dim1_mem)

##############################################################################
# Pure python implementations (reference) of the code in lib.cpp
##############################################################################

def _gower_matrix(X):
    assert X.ndim == 3
    assert X.shape[1] == 3, 'middle dimension must be three'

    B = sum(np.dot(x.T, x) for x in X) / float(len(X))   
    
    assert B.shape == (X.shape[2], X.shape[2])
    
    b = B.mean(1)
    bb = b.mean()
    return (B - np.add.outer(b, b)) + bb

def _center_inplace(X):
    assert X.shape[0] == 3
    muX, muY, muZ = X.mean(1)

    X[0,:] -= muX
    X[1,:] -= muY
    X[2,:] -= muZ


def _is_mirror_image(X, Y):
    assert X.shape[0] == 3, 'x.shape[0] != 3'
    assert Y.shape[0] == 3, 'x.shape[0] != 3'
    ## SVD of correlation matrix
    
    V, L, U = scipy.linalg.svd(np.dot(X, Y.T))
    R = np.dot(V, U)
    assert R.shape == (3,3)
    
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
    x = (U * np.sqrt(v)).T

    if reference is None:
        reference = X[-1]

    _center_inplace(x)
    _center_inplace(reference)
    
    x = _rectify_mirror(x, reference)
    if _is_mirror_image(x, reference):
        raise ValueError('It\'s a trap!')
    
    return x
    
def _rectify_mirror(X, Y):
    assert X.shape[0] == Y.shape[0] == 3, 'first dimension must be 3. you supplied X.shape=%s, Y.shape=%s' % (str(X.shape), str(Y.shape))
    
    i = 0
    while _is_mirror_image(X, Y) and i < 2:
        X[i] *= -1
        i += 1
    return X

##############################################################################
# Tests for each of the functions in lib.cpp, comparing it with the pure
# python
##############################################################################

def test_gower_matrix():
    "all of the assignments here are for the state k, so there's no skipping"
    n_frames = 1
    n_padded_atoms = 5
    n_real_atoms = 4

    cdef double[:, :, :] X = np.random.randn(n_frames, 3, n_padded_atoms)
    cdef long[:] assignments = np.zeros(n_frames, dtype=np.int64)
    cdef long k = 0

    _B = _gower_matrix(np.asarray(X)[:, :, 0:n_real_atoms])

    cdef double[:, :] B = np.zeros((n_real_atoms, n_real_atoms), dtype=np.double)
    
    gower_matrix(&X[0,0,0], X.shape[0], X.shape[1], n_real_atoms, X.shape[2],
                 &assignments[0], assignments.shape[0], k,
                 &B[0,0], B.shape[0], B.shape[1])
                 
    np.testing.assert_array_almost_equal(np.asarray(B), _B)
    print 'test_gower_matrix passed'


def test_gower_matrix2():
    "this tests the skipping of only states where assignments==k"
    n_frames = 10
    n_padded_atoms = 5
    n_real_atoms = 4

    cdef double[:, :, :] X = np.random.randn(n_frames, 3, n_padded_atoms)
    cdef np.ndarray[long, ndim=1] assignments = np.ones(n_frames, dtype=np.int64)
    assignments[0] = 0
    assignments[1] = 0
    cdef long k = 0

    _B = _gower_matrix(np.asarray(X)[assignments==k, :, 0:n_real_atoms])

    cdef double[:, :] B = np.zeros((n_real_atoms, n_real_atoms), dtype=np.double)
    
    gower_matrix(&X[0,0,0], X.shape[0], X.shape[1], n_real_atoms, X.shape[2],
                 &assignments[0], assignments.shape[0], k,
                 &B[0,0], B.shape[0], B.shape[1])
                 
    np.testing.assert_array_almost_equal(np.asarray(B), _B)
    print 'test_gower_matrix2 passed'


def test_center_inplace():
    "padded atoms == real atoms"
    cdef double[:, :] X = np.random.randn(3, 100)
    X2 = np.copy(np.asarray(X))
    _center_inplace(X2)

    center_inplace(&X[0,0], X.shape[0], X.shape[1],  X.shape[1])

    np.testing.assert_array_almost_equal(np.asarray(X), X2)
    print 'test_center_inplace passed'

def test_center_inplace2():
    "here there are more padded atoms than real atoms"
    n_atoms = 10
    n_padded_atoms = 12
    
    cdef np.ndarray[double, ndim=2] X = np.random.randn(3, n_padded_atoms)
    X2 = np.copy(np.asarray(X))
    _center_inplace(X2[:, 0:n_atoms])
     
    center_inplace(&X[0,0], X.shape[0], n_atoms, n_padded_atoms)
     
    np.testing.assert_array_almost_equal(np.asarray(X), X2)
    print 'test_center_inplace passed'



def test_is_mirror_image():
    cdef np.ndarray[double, ndim=2] X
    cdef np.ndarray[double, ndim=2] Y
    
    for i in range(1000):
        X = np.random.randn(3, 100)
        Y = np.random.randn(3, 100)
        _center_inplace(X)
        _center_inplace(Y)
    
        result1 = is_mirror_image(&X[0,0], X.shape[0], X.shape[1], X.shape[1],
                                  &Y[0,0], Y.shape[0], Y.shape[1], Y.shape[1])
        result2 = _is_mirror_image(X, Y)
        assert result1 == result2
    print 'test_is_mirror_image passed'


def test_is_mirror_image2():
    "more padded atoms than real atoms"
    n_atoms = 100
    n_padded_atoms = 110
    
    cdef np.ndarray[double, ndim=2] X
    cdef np.ndarray[double, ndim=2] Y
    
    for i in range(1000):
        X = np.random.randn(3, n_padded_atoms)
        Y = np.random.randn(3, n_padded_atoms)
        _center_inplace(X[:, :n_atoms])
        _center_inplace(Y[:, :n_atoms])
    
        result1 = is_mirror_image(&X[0,0], X.shape[0], n_atoms, n_padded_atoms,
                                  &Y[0,0], Y.shape[0], n_atoms, n_padded_atoms)
        result2 = _is_mirror_image(X[:, :n_atoms], Y[:, :n_atoms])
        assert result1 == result2
    print 'test_is_mirror_image passed'

def test_average_structure():
    
    n_frames = 10
    n_padded_atoms = 5
    n_real_atoms = 4

    cdef double[:, :, :] X = np.random.randn(n_frames, 3, n_padded_atoms)
    X2 = np.asarray(X).copy()
    cdef long[:] assignments = np.zeros(n_frames, dtype=np.int64)
    cdef long k = 0
    cdef double[:, :] result = np.zeros((3, n_real_atoms), dtype=np.double)

    average_structure(&X[0,0,0], X.shape[0], X.shape[1], n_real_atoms, X.shape[2],
                 &assignments[0], assignments.shape[0], k,
                 &result[0,0], result.shape[0], result.shape[1], result.shape[1])


    r2 = _average_structure(X2[:, :, 0:n_real_atoms])
    
    #print np.asarray(result)
    #print r2
    
    difference = rmsd_kabsch(np.asarray(result).T, r2.T)
    np.testing.assert_almost_equal(difference, 0)

    print 'test_average_structure passed'


def test_average_structure2():
    "more padded atoms than real atoms"
    n_frames = 10
    n_padded_atoms = 12
    n_real_atoms = 10

    cdef double[:, :, :] X = np.random.randn(n_frames, 3, n_padded_atoms)
    X2 = np.asarray(X).copy()
    cdef np.ndarray[long, ndim=1] assignments = np.ones(n_frames, dtype=np.int64)
    assignments[0] = 0
    assignments[1] = 0
    cdef long k = 0
    cdef double[:, :] result = np.zeros((3, n_real_atoms), dtype=np.double)

    average_structure(&X[0,0,0], X.shape[0], X.shape[1], n_real_atoms, X.shape[2],
                 &assignments[0], assignments.shape[0], k,
                 &result[0,0], result.shape[0], result.shape[1], result.shape[1])


    r2 = _average_structure(X2[assignments==k, :, 0:n_real_atoms])
    
    #print np.asarray(result)
    #print r2
    
    difference = rmsd_kabsch(np.asarray(result).T, r2.T)
    np.testing.assert_almost_equal(difference, 0)

    print 'test_average_structure2 passed'

def test_average_structure3():
    "like test_average_structure2, but now the results array also has padded atoms"
    n_frames = 10
    n_padded_atoms = 12
    n_real_atoms = 10

    cdef double[:, :, :] X = np.random.randn(n_frames, 3, n_padded_atoms)
    X2 = np.asarray(X).copy()
    cdef np.ndarray[long, ndim=1] assignments = np.ones(n_frames, dtype=np.int64)
    assignments[0] = 0
    assignments[1] = 0
    cdef long k = 0
    cdef double[:, :] result = np.zeros((3, n_padded_atoms), dtype=np.double)

    average_structure(&X[0,0,0], X.shape[0], X.shape[1], n_real_atoms, X.shape[2],
                 &assignments[0], assignments.shape[0], k,
                 &result[0,0], result.shape[0], n_real_atoms, n_padded_atoms)


    r2 = _average_structure(X2[assignments==k, :, 0:n_real_atoms])
    
    #print np.asarray(result)
    #print r2
    
    difference = rmsd_kabsch(np.asarray(result[:, 0:n_real_atoms]).T, r2.T)
    np.testing.assert_almost_equal(difference, 0)

    print 'test_average_structure2 passed'



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
       [  7.49631938e-02,   7.27274697e-01,   1.68029988e-01]]).T

    cdef np.ndarray[double, ndim=2] conf2 = np.array([[ -2.65586755e-01,  -9.27238597e-01,  -1.88187143e+00],
           [ -5.86940362e-01,  -1.01088117e+00,  -8.27745500e-04],
           [  4.84376667e-01,  -7.68769912e-01,   2.55097302e+00],
           [  8.75585443e-01,   6.06322965e-01,   3.11763150e-01],
           [ -1.49247711e-01,   1.64926174e-01,   4.32468497e-01],
           [  1.53470843e-01,  -4.55160757e-01,  -3.85001867e-01],
           [ -1.05591533e+00,   9.48715369e-01,   1.14833950e+00],
           [  3.71162052e-01,  -2.00374187e-01,  -6.39289609e-01],
           [  2.48058346e-01,   9.15185423e-01,  -1.70458350e+00],
           [ -7.49631938e-02,   7.27274697e-01,   1.68029988e-01]]).T

    cdef np.ndarray[double, ndim=2] reference = np.array([[ 0.82221491, -0.36405214,  1.05510652],
           [-0.11809122,  0.40392022,  1.44847452],
           [-0.55748556, -1.19532741, -0.03597826],
           [-0.24324392, -0.32296946,  1.48038213],
           [-0.13996595,  1.17421015, -0.39945472],
           [ 0.58941451, -0.42541861, -1.39731127],
           [ 2.63017262, -0.04900264, -0.11425032],
           [-0.22778791, -0.30019201, -0.6070062 ],
           [-1.20558585,  0.90675396, -0.8028707 ],
           [-1.54964163,  0.17207795, -0.62709169]]).T
           
    conf1 = np.ascontiguousarray(conf1)
    conf2 = np.ascontiguousarray(conf2)
    reference = np.ascontiguousarray(reference)
    

    cdef np.ndarray[double, ndim=2] conf1_copy = np.copy(conf1)
    _rectify_mirror(conf1_copy, reference)
    

    cdef np.ndarray[double, ndim=2] conf1_copy2 = np.copy(conf1)
    rectify_mirror(&conf1_copy2[0,0], conf1_copy2.shape[0], conf1_copy2.shape[1], conf1_copy2.shape[1],
                   &reference[0,0], reference.shape[0], reference.shape[1], reference.shape[1])
        
    np.testing.assert_array_almost_equal(conf1_copy2, conf1_copy)
    print 'test_mirror_flipper passed'
