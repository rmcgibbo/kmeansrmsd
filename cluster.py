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
RMSD KMeans clustering. Command line script. (run with -h for details)
"""
##############################################################################
# Imports
##############################################################################

import os
import warnings
import pprint
import yaml
import numpy as np
import tables
from argparse import ArgumentParser

from mdtraj import io
from msmbuilder.clustering import split
from kmeansrmsd.medoids import _hybrid_kmedoids as medoids
from kmeansrmsd.pyclustering import kmeans_mds as pykmeans_mds

try:
    from kmeansrmsd.clustering import kmeans_mds as ckmeans_mds
    HAVE_C_CODE = True
except ImportError:
    ckmeans_mds = lambda *args: log("C implementation is not available")
    HAVE_C_CODE = False


##############################################################################
# Functions
##############################################################################

def main():
    args, atom_indices, project, project_root = parse_cmdline()

    # load all of the data from disk
    xyzlist, traj_lengths = load_trajs(project, os.path.dirname(args.project_yaml),
                                       atom_indices, args.stride)

    n_real_atoms = len(atom_indices)
    n_padded_atoms = xyzlist.shape[2]
    assert n_padded_atoms >= n_real_atoms

    if args.implementation == 'c':
        log('clustering: kmeans (C)')
        if args.nearest_medoid:
            raise NotImplementedError('nearest_medoid is only implemented on the'
                                      'pure python platform (-i py)')
        results = ckmeans_mds(xyzlist, k=args.n_clusters, n_real_atoms=n_real_atoms,
                             max_time=args.max_time, max_iters=args.max_iters,
                             threshold=args.epsilon)
        centers, assignments, distances, scores, times = results

    elif args.implementation == 'py':
        log('clustering: kmeans (py)')

        # don't mess around with the mess that is padding atoms. just copy
        # the data into this view, and then let numpy deal with it.
        xyzlist = xyzlist.swapaxes(1,2)
        xyzlist = xyzlist[:, 0:n_real_atoms, :]

        results = pykmeans_mds(xyzlist, k=args.n_clusters, max_iters=args.max_iters,
                               max_time=args.max_time, threshold=args.epsilon,
                               nearest_medoid=args.nearest_medoid)
        centers, assignments, distances, scores, times = results

    elif args.implementation == 'medoids':
        log('clustering: k medoids')
        if args.nearest_medoid:
            raise NotImplementedError('nearest_medoid is only for the kmeans code')

        from msmbuilder.metrics import RMSD
        metric = RMSD()
        # again, let's not deal with the padding atoms. we'll let the msmbuilder TheoData
        # deal with this. this is inefficient w/ memory, but easier to get right
        # for testing.
        xyzlist = xyzlist.swapaxes(1,2)
        ptraj = metric.TheoData(xyzlist[:, 0:n_real_atoms, :])
        results = medoids(metric, ptraj, k=args.n_clusters, num_iters=args.max_iters,
                          local_swap=True, too_close_cutoff=0.0001, ignore_max_objective=True)
        centers, assignments, distances, scores, times = results

    save(args.output, traj_lengths, args.stride, n_real_atoms, centers, assignments, distances, scores, times)


def log(msg, *args):
    "log a string to stdout (i.e just print it). lightweight."
    if len(args) > 0:
        msg = str(msg) % args
    print msg


def _convert_from_lossy_integers(X, precision=1000, dtype=np.float32):
    """Implementation of the lossy compression used in Gromacs XTC using
    the pytables library. Convert 16 bit integers into 32 bit floats."""
    X2 = X.astype(dtype)
    X2 /= float(precision)
    return X2


def load_trajs(project, project_root, atom_indices, stride):
    """Load trajectories from the projectfile into xyzlist

    This loads the trajectories in atom major order, and also puts in
    padding atoms (coordinates at zero) so that the number of atoms is a multiple
    of four.

    Parameters
    ----------
    project : dict
        The result of loading the msmbuilder project file (yaml).
    project_root : str
        path to the directory on disk where the project file lives. this
        is used as the root directory from which to find the trajectories
        locations on disk.
    atom_indices : np.ndarray, dtype=int
        The indices of the atoms that you want to use for the RMSD computation.
        We only load those ones.

    Returns
    -------
    xyzlist : np.ndarray, dtype=np.float64, shape=(n_frames, 3, n_padded_atoms)
        The coordinates of all of the atoms.
    """
    def length_after_stride(length, stride):
        """If we have an array of length `length`, and we stride it by `stride`,
        how long will it be? Equivalent to len(range(length)[::stride])
        """
        return length // stride + bool(length % stride)

    trajs = [t for t in project['trajs'] if t['errors'] is None]
    assert len(trajs) > 0, 'no trajs found in project file'

    n_frames = sum(length_after_stride(t['length'], stride) for t in trajs)
    n_real_atoms = len(atom_indices)
    # for rmsd, the number of atoms must be a multiple of four
    n_padded_atoms = 4 + n_real_atoms - n_real_atoms % 4

    # we're asking for float64 here because currently the kmeans code require double
    # (it will make a single precision copy for the RMSD code itself)
    log('allocating array of size %.1f MB...', n_frames*3*n_padded_atoms*8 / (1024.0**2))
    xyzlist = np.zeros((n_frames, 3, n_padded_atoms), dtype=np.float64)

    log('loading trajectories...')
    traj_lengths = np.empty(len(trajs), dtype=np.int)
    traj_lengths.fill(-1)

    xyzlist_pointer = 0
    for i, traj_record in enumerate(trajs):
        path = os.path.join(project_root, traj_record['path'])
        with tables.File(path) as f:
            # xyz contains the trajectory data from this file
            # we reshape it into axis major ordering
            xyz = f.root.XYZList[::stride, atom_indices, :].swapaxes(1,2)
            traj_lengths[i] = len(xyz)

        assert xyz.shape[1] == 3, 'not in axis major ordering'
        if xyz.dtype == np.int16:
            xyz = _convert_from_lossy_integers(xyz, dtype=np.float64)

        if np.any(np.logical_or(np.isinf(xyz), np.isnan(xyz))):
            raise ValueError('There are infs or nans in the trajectory loaded from %s' % path)
        if np.max(np.abs(xyz)) > 32:
            warnings.warn('max coordinate (%f) in %s is greater than 32nm from origin. methinks something is wrong?' % (np.max(xyz), path))

        # and accumulate the data into xyzlist
        xyzlist[xyzlist_pointer:xyzlist_pointer+len(xyz), :, 0:n_real_atoms] = xyz
        xyzlist_pointer += len(xyz)

    assert xyzlist_pointer == len(xyzlist), 'shape error when loading stride probably?'
    return xyzlist, traj_lengths


def parse_cmdline():
    parser = ArgumentParser('k-means RMSD clustering')

    parser.add_argument('-k', '--n_clusters', required=True, type=int, help='''
        Number of clusters to find. Using more clusters gives a finer partitioning
        of phase space which reduces the systematic error in the markov model, but
        this comes at the cost of statistical error (variance) in the estimation
        of the transition matrix. Also, it is computationally more costly to find
        more clusters. The running time of this algorithm is linear in the number
        of clusters.''')

    convergence_group = parser.add_argument_group('convergence criteria', '''The algorithm
        will terminate the first time one of the convergence criteria below trips.''')
    convergence_group.add_argument('-e', '--epsilon', default=1e-8, type=float,
        help='''When the root-mean-square radius of the states (in nm) decreases
        less than this value during a round.''')
    convergence_group.add_argument('-t', '--max_time', default=None, type=int,
        help='''After this amount of time has elapsed (in seconds).''')
    convergence_group.add_argument('-n', '--max_iters', default=100, type=int,
        help='''After this number of iterations.''')

    implementation_choices = ['py', 'medoids']
    if HAVE_C_CODE:
        implementation_choices.append('c')
            
    parser.add_argument('-i', '--implementation', choices=implementation_choices,
        default='c', help='''Which algorithm / implementation do you want? "c"
        corresponds to the kmeans rmsd algorithm implemented in C; "py" corresponds
        to the same kmeans algorithm implemented in python (useful for reference,
        testing and performance benchmarking); and "medoids" corresponds to the
        swap-based kmedoids algorithm from msmbuilder 2.5.''')
    parser.add_argument('-a', '--atom_indices', required=True, help='''Path to a
        plain text file listing the indices of the atoms to use for the RMSD
        calculation. The file should be zero indexed.''')
    parser.add_argument('-s', '--stride', type=int, default=1, help='''Load only
        every n-th frame of simulation data, effectively reducing your amount
        of data by `stride`-th times. This can be useful to save memory and
        increase the computation speed, at the cost of some loss in accuracy.''')
    parser.add_argument('-p', '--project_yaml', required=True, help='''Path
        to the msmbuilder project description file, in yaml format. This file
        lists the paths to all of your trajectories on disk. Curently, only the
        HDF5 format trajectories are accepted, because we use some nontrivial
        sliceing and striding that is a bit of a pain to implement with the
        order format readers.''')
    parser.add_argument('-m', '--nearest_medoid', action='store_true', help='''
        Somewhat hacky option, specific to the pure python implementation (-i 
        py). After computing the mean via the MDS averaging procedure, we
        instead choose as the center the data point in that grouping closest
        to the MDS structure. So this is really a medoids type approach.''')


    output = parser.add_argument_group('output')
    output.add_argument('-o', '--output', required=True, default='Data/',
        help='''path to output directory.''')
    output.add_argument('-f', '--force', default=False, action='store_true',
        help='Overwrite the output directory if it exists.')

    args = parser.parse_args()
    if os.path.exists(args.output):
        if args.force:
            os.unlink(args.output)
        else:
            raise IOError('output directory %s already exists. use something different?' % args.output)

    log(pprint.pformat(args.__dict__))
    project_root = os.path.dirname(args.project_yaml)

    log('opening project file...')
    with open(args.project_yaml) as f:
        project = yaml.load(f)

    atom_indices = np.loadtxt(args.atom_indices, int)

    return args, atom_indices, project, project_root


def reshape_for_output(array, dtype, traj_lengths, stride):
    output = np.empty((len(traj_lengths), stride*np.max(traj_lengths)), dtype)
    output.fill(np.nan)

    array2 = split(array, traj_lengths)
    for i, row in enumerate(array2):
        assert len(row) == traj_lengths[i], 'reshape error'
        output[i, :len(row)*stride:stride] = row
    return output


def save(outdir, traj_lengths, stride, n_real_atoms,
         centers, assignments, distances, scores, times):

    assignments = reshape_for_output(assignments, np.int, traj_lengths, stride)
    distances = reshape_for_output(distances, np.float, traj_lengths, stride)

    centers = centers.swapaxes(1,2)
    centers = centers[:, 0:n_real_atoms, :]

    os.makedirs(outdir)
    log('saving results to %s/' % outdir)
    io.saveh(os.path.join(outdir, 'centers.h5'), XYZList=centers)
    io.saveh(os.path.join(outdir, 'Assignments.h5'), assignments)
    io.saveh(os.path.join(outdir, 'Assignments.h5.distances'), distances)
    if len(scores) > 0 and len(times) > 0:
        io.saveh(os.path.join(outdir, 'convergence.h5'), scores=scores, times=times)


if __name__ == '__main__':
    main()
