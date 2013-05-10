from kmeansrmsd import test
from kmeansrmsd.clustering import kmeans_mds
test.test()

import numpy as np
import pyrr

def test_kmeans():
    dataset = []
    for i in range(10):
        xyz = np.random.randn(12,3)
        for j in range(100):
            rot = pyrr.matrix33.create_from_quaternion(pyrr.quaternion.create_from_eulers(np.random.rand(3)*2*np.pi))
            dataset.append(np.dot(xyz + 0.01*np.random.randn(*xyz.shape), rot).T)
    dataset = np.array(dataset)
    dataset[:, :, 0] = 0


    centers, assignments, distances, scores, times = kmeans_mds(dataset, k=10)
    print assignments

test_kmeans()