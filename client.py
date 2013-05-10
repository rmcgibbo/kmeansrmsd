from kmeansrmsd import test
from kmeansrmsd.clustering import kmeans_mds
test.test()

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
