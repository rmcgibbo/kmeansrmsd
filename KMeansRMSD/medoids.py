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

import numpy as np
import sys
import time

def _kcenters(metric, ptraj, k=None, distance_cutoff=None, seed=0, verbose=True):
    """Run kcenters clustering algorithm.

    Terminates either when `k` clusters have been identified, or when every data
    is clustered better than `distance_cutoff`.

    Parameters
    ----------
    metric : msmbuilder.metrics.AbstractDistanceMetric
        A metric capable of handling `ptraj`
    ptraj : prepared trajectory
        ptraj return by the action of the preceding metric on a msmbuilder trajectory
    k : {int, None}
        number of desired clusters, or None
    distance_cutoff : {float, None}
        Stop identifying new clusters once the distance of every data to its
        cluster center falls below this value. Supply either this or `k`
    seed : int, optional
        index of the frame to use as the first cluster center
    verbose : bool, optional
        print as each new generator is found

    Returns
    -------
    generator_indices : ndarray
        indices (with respect to ptraj) of the frames to be considered cluster centers
    assignments : ndarray
        the cluster center to which each frame is assigned to (1D)
    distances : ndarray
        distance from each of the frames to the cluster center it was assigned to

    See Also
    --------
    KCenters : wrapper around this implementation that provides more convenience

    Notes
    ------
    the assignments are numbered with respect to the position in ptraj of the
    generator, not the position in generator_indices. That is, assignments[10] =
    1020 means that the 10th simulation frame is assigned to the 1020th
    simulation frame, not to the 1020th generator.

    References
    ----------
    .. [1] Beauchamp, MSMBuilder2
    """


    if k is None and distance_cutoff is None:
        raise ValueError("I need some cutoff criterion! both k and distance_cutoff can't both be none")
    if k is None and distance_cutoff <= 0:
        raise ValueError("With k=None you need to supply a legit distance_cutoff")
    if distance_cutoff is None:
        # set it below anything that can ever be reached
        distance_cutoff = -1
    if k is None:
        # set k to be the highest 32bit integer
        k = sys.maxint

    distance_list = np.inf * np.ones(len(ptraj), dtype=np.float32)
    assignments = -1 * np.ones(len(ptraj), dtype=np.int32)

    generator_indices = []
    for i in xrange(k):
        new_ind = seed if i == 0 else np.argmax(distance_list)
        #if k == sys.maxint:
        #    print "K-centers: Finding generator %i. Will finish when % .4f drops below % .4f" % (i, distance_list[new_ind], distance_cutoff)
        #else:
        #    print "K-centers: Finding generator %i" % i

        if distance_list[new_ind] < distance_cutoff:
            break
        new_distance_list = metric.one_to_all(ptraj, ptraj, new_ind)
        updated_indices = np.where(new_distance_list < distance_list)[0]
        distance_list[updated_indices] = new_distance_list[updated_indices]
        assignments[updated_indices] = new_ind
        generator_indices.append(new_ind)

    if verbose:
        print 'KCenters found %d generators' % (i + 1)

    return np.array(generator_indices), assignments, distance_list

    
def _hybrid_kmedoids(metric, ptraj, k=None, distance_cutoff=None, num_iters=10, local_swap=True, too_close_cutoff=0.0001, ignore_max_objective=False, initial_medoids='kcenters', initial_assignments=None, initial_distance=None):
    """Run the hybrid kmedoids clustering algorithm to cluster a trajectory

    References
    ----------
    .. [1] Beauchamp, K. MSMBuilder2

    Parameters
    ----------
    metric : msmbuilder.metrics.AbstractDistanceMetric
        A metric capable of handling `ptraj`
    ptraj : prepared trajectory
        ptraj return by the action of the preceding metric on a msmbuilder trajectory
    k : int
        number of desired clusters
    num_iters : int
        number of swaps to attempt per medoid
    local_swap : boolean, optional
        If true, proposed swaps will be between a medoid and a data point
        currently assigned to that medoid. If false, the data point for the
        proposed swap is selected randomly.
    too_close_cutoff : float, optional
        Summarily reject proposed swaps if the distance of the medoid to the trial
        medoid is less than thus value
    ignore_max_objective : boolean, optional
        Ignore changes to the distance of the worst classified point, and only
        reject or accept swaps based on changes to the p norm of all the data
        points.
    initial_medoids : {'kcenters', ndarray}
        If 'kcenters', run kcenters clustering first to get the initial medoids,
        and then run the swaps to improve it. If 'random', select the medoids at
        random. Otherwise, initial_medoids should be a numpy array of the
        indices of the medoids.
    initial_assignments : {None, ndarray}, optional
        If None, initial_assignments will be computed based on the
        initial_medoids. If you pass in your own initial_medoids, you can also
        pass in initial_assignments to avoid recomputing them.
    initial_distances : {None, ndarray}, optional
        If None, initial_distances will be computed based on the initial_medoids.
        If you pass in your own initial_medoids, you can also pass in
        initial_distances to avoid recomputing them.

    """
    if k is None and distance_cutoff is None:
        raise ValueError("I need some cutoff criterion! both k and distance_cutoff can't both be none")
    if k is None and distance_cutoff <= 0:
        raise ValueError("With k=None you need to supply a legit distance_cutoff")
    if distance_cutoff is None:
        # set it below anything that can ever be reached
        distance_cutoff = -1

    num_frames = len(ptraj)
    if initial_medoids == 'kcenters':
        initial_medoids, initial_assignments, initial_distance = _kcenters(metric, ptraj, k, distance_cutoff)
    elif initial_medoids == 'random':
        if k is None:
            raise ValueError('You need to supply the number of clusters, k, you want')
        initial_medoids = np.random.permutation(np.arange(num_frames))[0:k]
        initial_assignments, initial_distance = _assign(metric, ptraj, initial_medoids)
    else:
        if not isinstance(initial_medoids, np.ndarray):
            raise ValueError('Initial medoids should be a numpy array')
        if initial_assignments is None or initial_distance is None:
            initial_assignments, initial_distance = _assign(metric, ptraj, initial_medoids)

    assignments = initial_assignments
    distance_to_current = initial_distance
    medoids = initial_medoids
    pgens = ptraj[medoids]
    k = len(initial_medoids)
    times = [] # ADDED
    scores = []

    obj_func = np.sqrt(np.mean(np.square(distance_to_current)))
    max_norm = np.max(distance_to_current)

    for iteration in xrange(num_iters):
        print 'Outer iteration %d: RMS radius %.5f' % (iteration, obj_func)
        for medoid_i in xrange(k):

            if local_swap is False:
                trial_medoid = np.random.randint(num_frames)
            else:
                trial_medoid = np.random.choice(np.where(assignments == medoids[medoid_i])[0])

            old_medoid = medoids[medoid_i]

            if old_medoid == trial_medoid:
                continue

            new_medoids = medoids.copy()
            new_medoids[medoid_i] = trial_medoid
            pmedoids = ptraj[new_medoids]

            new_distances = distance_to_current.copy()
            new_assignments = assignments.copy()

            # print 'Sweep %d, swapping medoid %d (conf %d) for conf %d...' % (iteration, medoid_i, old_medoid, trial_medoid)

            distance_to_trial = metric.one_to_all(ptraj, ptraj, trial_medoid)
            if distance_to_trial[old_medoid] < too_close_cutoff:
                print 'Too close'
                continue

            assigned_to_trial = np.where(distance_to_trial < distance_to_current)[0]
            new_assignments[assigned_to_trial] = trial_medoid
            new_distances[assigned_to_trial] = distance_to_trial[assigned_to_trial]

            ambiguous = np.where((new_assignments == old_medoid) & \
                                 (distance_to_trial >= distance_to_current))[0]
            for l in ambiguous:
                d = metric.one_to_all(ptraj, pmedoids, l)
                argmin = np.argmin(d)
                new_assignments[l] = new_medoids[argmin]
                new_distances[l] = d[argmin]

            new_obj_func = np.sqrt(np.mean(np.square(new_distances)))
            new_max_norm = np.max(new_distances)

            if new_obj_func < obj_func and (new_max_norm <= max_norm or ignore_max_objective is True):
                #print "Accept. New f = %f, Old f = %f" % (new_obj_func, obj_func)
                medoids = new_medoids
                assignments = new_assignments
                distance_to_current = new_distances
                obj_func = new_obj_func
                max_norm = new_max_norm
            else:
                pass
                #print "Reject. New f = %f, Old f = %f" % (new_obj_func, obj_func)
            scores.append(obj_func)
            times.append(time.time())

    return medoids, assignments, distance_to_current, np.array(scores), np.array(times)