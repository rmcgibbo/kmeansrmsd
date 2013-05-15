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
Silhouette score (unsupervised metric) for clustering. Run this script with -h
for details
"""
##############################################################################
# Imports
##############################################################################

import os
import pprint
import yaml
import numpy as np
import tables
from argparse import ArgumentParser

from sklearn.metrics.cluster import silhouette_score

from mdtraj import io
from mdtraj import IRMSD
from kmeansrmsd.clustering import sremove_center_of_mass, scalculate_g

##############################################################################
# Functions
##############################################################################

def main():
    args, atom_indices, project, project_root = parse_cmdline()

    # load all of the data from disk
    xyzlist, sampled_frames = load_trajs(project, os.path.dirname(args.project_yaml),
                                       atom_indices, args.stride, args.fraction)
    assignments = io.loadh(args.assignments, 'arr_0')
    # pick only the assignments that had their xyz data loaded
    assignments = np.concatenate([assignments[i, sampled_frames[i]] for i in range(len(sampled_frames))])

    # make sure we didn't mess up the subsampling and get nonsense data
    assert not np.any(assignments < 0), 'assignments negative? stride/sampling messed up probs. did you use a different strid than you clustered with?'
    #assert np.all(np.unique(assignments) == np.arange(np.max(assignments)+1)), 'assignments dont go from 0 to max. did you use a different strid than you clustered with?'

    n_real_atoms = len(atom_indices)
    n_padded_atoms = xyzlist.shape[2]
    assert n_padded_atoms >= n_real_atoms

    pairwise = calculate_pairwise_rmsd(xyzlist, n_real_atoms)

    print 'computing silhouette...'
    score = silhouette_score(pairwise, assignments, metric='precomputed')
    print 'silhouette score: %f' % score

    path = os.path.join(args.output, 'silhouette.dat')
    print 'saving results to flat text file (append): %s...' % path
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    with open(path, 'a') as f:
        f.write('%f\n' % score)


def calculate_pairwise_rmsd(xyzlist, n_real_atoms):
    n_frames = len(xyzlist)
    log('pairwise distance matrix will be %.2f MB...', 4*n_frames**2 / (1024.0**2))
    pairwise_distance = np.empty((n_frames, n_frames), dtype=np.float32)

    sremove_center_of_mass(xyzlist, n_real_atoms)
    g = scalculate_g(xyzlist)

    for i in range(n_frames):
        if i % 100 == 0:
            print '%d/%d' % (i, n_frames)
        pairwise_distance[i, :] = IRMSD.rmsd_one_to_all(xyzlist, xyzlist, g, g, n_real_atoms, i)

    return pairwise_distance


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


def load_trajs(project, project_root, atom_indices, stride, fraction):
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
    stride : int
    fraction : float


    Returns
    -------
    xyzlist : np.ndarray, dtype=np.float64, shape=(n_frames, 3, n_padded_atoms)
        The coordinates of all of the atoms.
    """
    trajs = [t for t in project['trajs'] if t['errors'] is None]

    # these are the indices to take from each traj based on striding
    which_frames = [range(t['length'])[::stride] for t in trajs]

    # now further reduce the amount by taking a random sampling of them.
    sampled_frames = [np.random.permutation(w)[:int(len(w)*fraction)] for w in which_frames]
    n_frames = sum(len(s) for s in sampled_frames)

    n_real_atoms = len(atom_indices)
    # for rmsd, the number of atoms must be a multiple of four
    n_padded_atoms = 4 + n_real_atoms - n_real_atoms % 4

    # we're asking for float64 here because currently the kmeans code require double
    # (it will make a single precision copy for the RMSD code itself)
    log('allocating space for trajectories of size %.1f MB...', n_frames*3*n_padded_atoms*4 / (1024.0**2))
    xyzlist = np.zeros((n_frames, 3, n_padded_atoms), dtype=np.float32)

    log('loading trajectories...')

    xyzlist_pointer = 0
    for i, traj_record in enumerate(trajs):
        path = os.path.join(project_root, traj_record['path'])
        with tables.File(path) as f:
            # xyz contains the trajectory data from this file
            # we reshape it into axis major ordering
            xyz = f.root.XYZList[sampled_frames[i], :, :]
            xyz = xyz[:, atom_indices, :].swapaxes(1, 2)

        assert xyz.shape[1] == 3, 'not in axis major ordering'

        if xyz.dtype == np.int16:
            xyz = _convert_from_lossy_integers(xyz, dtype=np.float32)

        if np.any(np.logical_or(np.isinf(xyz), np.isnan(xyz))):
            raise ValueError('There are infs or nans in the trajectory loaded from %s' % path)
        if np.max(np.abs(xyz)) > 32:
            warnings.warn('max coordinate (%f) in %s is greater than 32nm from origin. methinks something is wrong?' % (np.max(xyz), path))

        # and accumulate the data into xyzlist
        xyzlist[xyzlist_pointer:xyzlist_pointer+len(xyz), :, 0:n_real_atoms] = xyz
        xyzlist_pointer += len(xyz)

    assert xyzlist_pointer == len(xyzlist), 'shape error when loading stride probably?'
    return xyzlist, sampled_frames


def parse_cmdline():
    parser = ArgumentParser(__file__, description='''
    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (``a``) and the mean nearest-cluster distance (``b``) for each
    sample.  The Silhouette Coefficient for a sample is ``(b - a) / max(a,
    b)``.  To clarify, ``b`` is the distance between a sample and the nearest
    cluster that the sample is not a part of.''')
    parser.add_argument('-p', '--project_yaml', required=True, help='''Path
        to the msmbuilder project description file, in yaml format. This file
        lists the paths to all of your trajectories on disk. Curently, only the
        HDF5 format trajectories are accepted, because we use some nontrivial
        sliceing and striding that is a bit of a pain to implement with the
        order format readers.''')
    parser.add_argument('-s', '--stride', required=True, default=1, type=int,
        help='''Computing the silhouette score requires knowing the assignments,
        which, due to striding at the clustering level, we only know for every
        n-th point. You need to thus supply the same stride here that you did
        for the clustering step, otherwise the assignments wont match up with
        the loaded data. IMPORTANT''')
    parser.add_argument('-r', '--fraction', required=True, default=1, type=float,
        help='''Subsampling fraction. The computation of the silhouette requires
        getting the full pairwise RMSD matrix between every conformation, which
        is very memory intensive, even with striding. Use this option to reduce
        the data size by this fraction.''')
    parser.add_argument('-a', '--atom_indices', required=True, help='''Path to a
        plain text file listing the indices of the atoms to use for the RMSD
        calculation. The file should be zero indexed.''')
    parser.add_argument('-g', '--assignments', required=True)
    parser.add_argument('-o', '--output', required=True, default='Data/', help='''
        path to output directory. the output of this calculation is a single
        number, and it will be appended to the flat text file "silhouette.dat" in the
        output directory''')

    args = parser.parse_args()

    log(pprint.pformat(args.__dict__))
    project_root = os.path.dirname(args.project_yaml)

    log('opening project file...')
    with open(args.project_yaml) as f:
        project = yaml.load(f)

    atom_indices = np.loadtxt(args.atom_indices, int)

    return args, atom_indices, project, project_root


if __name__ == '__main__':
    main()
