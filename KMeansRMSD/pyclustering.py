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
Pure python implementation of the kmeans rmsd algorithm. This code is slightly
slower than the C version, but not *that* much slower. In both cases, the RMSD
implementation is still in C, and that's ususally more costly that the update
step. But it is still a significant improvement. This is the best place to
start to understand the code though.
"""

###############################################################
# Imports
###############################################################

import time
import itertools

import numpy as np
import scipy.linalg

from mdtraj import IRMSD
from mdtraj.geometry import rmsd
from mdtraj.utils.arrays import ensure_type

__all__ = ['kmeans_mds']

###############################################################
# Functions
###############################################################

def kmeans_mds(xyzlist, k=10, max_iters=100, max_time=10, threshold=1e-8):
    """k-means clustering with the RMSD distance metric.

    this is an iterative algorithm. during each iteration we first move each cluster center to
    the empirical average of the conformations currently assigned to it, and then we re-assign
    all of the conformations given the new locations of the centers.
    
    to compute the average conformations, we use a form of classical multidimensional
    scaling / principle coordinate analysis.
    """
    xyzlist = ensure_type(xyzlist, np.float32, 3, name='xyzlist', shape=(None, None, 3), warn_on_cast=False)
    
    # center
    for x in xyzlist:
        centroid = x.astype('float64').mean(0)
        assert centroid.shape == (3,)
        x -= centroid

    # setup for the rmsd calculation
    n_frames, n_atoms = xyzlist.shape[0:2]
    xyzlist_irmsd, n_atoms_padded = rmsd.reshape_irmsd(xyzlist)
    xyzlist_G = rmsd.calculate_G(xyzlist)
    
    # setup for the clustering stuff
    # assignments[i] = j means that the i-th conformation is assigned to the j-th cluster
    assignments = -1*np.ones(n_frames, dtype=np.int64)
    assignments[0:k] = np.arange(k)
    np.random.shuffle(assignments)
    
    # the j-th cluster has cartesian coorinates centers[j]
    centers = np.zeros((k, xyzlist.shape[1], 3))
    # assignment_dist[i] gives the RMSD between the ith conformation and its
    # cluster center
    assignment_dist = np.inf * np.ones(len(xyzlist))
    # previous value of the clustering score
    # all of the clustering scores
    scores = [np.inf]
    times = [time.time()]
    
    for n in itertools.count():
        # recenter each cluster based on its current members
        for i in range(k):
            structures = xyzlist[assignments == i, :, :]
            if len(structures) == 0:
                # if the current state has zero assignments, just randomly
                # select a structure for it
                print 'warning: cluster %5d contains zero structures, reseeding...' % i
                print '(if this error appears once or twice at the beginning and then goes away'
                print 'don\'t worry. but if it keeps up repeatedly, something is wrong)'
                new_center = xyzlist[np.random.randint(len(xyzlist))]
            else:
                new_center = average_structure(structures)
            new_center -= new_center.mean(0)  # make sure the centers are always centered

            centers[i] = new_center
        
        # prepare the new centers for RMSD
        centers_G = rmsd.calculate_G(centers)
        centers_irmsd, _ = rmsd.reshape_irmsd(centers)
        
        # reassign all of the data
        assignments = -1 * np.ones(len(xyzlist))
        assignment_dist = np.inf * np.ones(len(xyzlist))
        for i in range(k):
            d = IRMSD.rmsd_one_to_all(centers_irmsd, xyzlist_irmsd, centers_G, xyzlist_G, n_atoms, i)
            where = d < assignment_dist
            assignments[where] = i
            assignment_dist[where] = d[where]

        # check how far each cluster center moved during the last iteration
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
    
    print 'RMSD KMeans Performance Summary (py)'
    print '------------------------------------'
    print 'n frames: %d' % n_frames
    print 'n states: %d' % k
    print 'mean time per round (s)   %.4f' % np.mean(np.diff(times))
    print 'stddev time per round (s) %.4f' % np.std(np.diff(times))
    print 'total time (s)            %.4f' % (times[-1] - times[0])
    return centers, assignments, assignment_dist, np.array(scores), np.array(times)


def gower_matrix(X):
    """
    Gower, J.C. (1966). Some distance properties of latent root
    and vector methods used in multivariate analysis.
    Biometrika 53: 325-338

    Parameters
    ----------
    X : np.ndarray, dtype=float, shape(n_frames, n_atoms, 3)
        The cartesian coordinates of each frame in an ensemble

    Returns
    --------
    B : np.ndarray, dtype=float, shape=(n_atoms, n_atoms)
        symmetric dissimilarity matrix, giving an average dissimilary
        of atom i to atom j over the ensemble

    Notes
    -----
    This code was adapted from the CSB toolbox (MIT License), https://csb.codeplex.com/
    """
    B = sum(np.dot(x, x.T) for x in X) / float(len(X))
   
    b = B.mean(1)
    bb = b.mean()

    return (B - np.add.outer(b, b)) + bb


def average_structure(X):
    """Calculate an average structure from an ensemble of structures

    Parameters
    ----------
    X : np.ndarray, dtype=float, shape(n_frames, n_atoms, 3)
        The cartesian coordinates of each frame in an ensemble
   
    Returns
    -------
    coordinates : np.ndarray, dtype=float, shape=(n_atoms, 3)
        The cartesian coordinates of an "average structure" of the ensemble
        
    Notes
    -----
    This code was adapted from the CSB toolbox (MIT License), https://csb.codeplex.com/
    """
    B = gower_matrix(X.astype(np.float64))
    v, U = scipy.linalg.eigh(B)
    if np.iscomplex(v).any():
        v = v.real
    if np.iscomplex(U).any():
        U = U.real

    indices = np.argsort(v)[-3:]
    v = np.take(v, indices, 0)
    U = np.take(U, indices, 1)
       
    x = U * np.sqrt(v)
    i = 0
    while is_mirror_image(x, X[0]) and i < 2:
        x[:, i] *= -1
        i += 1
    return x

            
def is_mirror_image(X, Y):
    """Check if two configurations X and Y are mirror images
    (i.e. their optimal superposition involves a reflection).

    Parameters
    ----------
    X : np.ndarray, dtype=float, shape(n_atoms, 3)
        The cartesian coordinates of one frame
    Y : np.ndarray, dtype=float, shape(n_atoms, 3)
        The cartesian coordinates of another frame
    
    Returns
    -------
    is_mirror_image : bool
        Are the two structures mirror images of one another?
    
    Notes
    -----
    This code was adapted from the CSB toolbox (MIT License), https://csb.codeplex.com/
    """
   
    ## center configurations

    X = X - np.mean(X, 0)
    Y = Y - np.mean(Y, 0)

    ## SVD of correlation matrix
    V, L, U = scipy.linalg.svd(np.dot(X.T, Y))
    R = np.dot(V, U)
    return scipy.linalg.det(R) < 0

