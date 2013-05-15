from __future__ import division
import numpy as np
import networkx as nx
import scipy.stats
import scipy.spatial
import scipy.sparse.linalg
import matplotlib.pyplot as pp


def graphcentrality(xyzlist, percentile):
    """Average an ensemble of structures with a diffusion-map approach
    on different atoms individually

    xyzlist : np.ndarray, shape(n_frames, n_atoms, 3)
        The cartesian coordinates in atom major order
    """
    n_frames, n_atoms, n_spatial = xyzlist.shape
    assert n_spatial == 3, 'n_spatial must be three'
    center = np.empty((n_atoms, n_spatial))

    if n_frames == 1:
        return xyzlist[0]

    for i in xrange(n_atoms):
        # coordinates of this atom in every frame
        x = xyzlist[:, i, :]

        # distance, for this atom, between every frame and every other frame
        d = scipy.spatial.distance_matrix(x, x)

        # compute the cutoff for binarization by taking the score
        # at this percentile, so that only `percentile` of the resulting
        # binarized version are true
        cutoff = scipy.stats.scoreatpercentile(d.reshape(-1), percentile)
        A = np.array(d < cutoff, np.float)
                
        # compute the first eigenvector of the resulting adjacency matrix
        w, v = scipy.sparse.linalg.eigsh(A, k=1)
        center_i = np.argmax(v)

        #print center_i
        center[i, :] = x[center_i, :]
        #center[i, :] = x[0, :]

    #return xyzlist[0]
    return center


def setup(n_frames=10):
    theta = sorted(np.random.rand(n_frames))
    x, y = np.cos(theta), np.sin(theta)
    xyzlist = np.empty((n_frames, 1, 2))
    xyzlist[:, 0, 0] = x
    xyzlist[:, 0, 1] = y
    xyzlist += 0.01 * np.random.randn(*xyzlist.shape)
    return xyzlist

def main():
    xyzlist = setup(500)
    
    for i in [50]:
        graphcentrality(xyzlist, percentile=i)

if __name__ == '__main__':
    main()
