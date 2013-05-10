import os
import sys
import yaml
import numpy as np
import tables
from kmeansrmsd.clustering import kmeans_mds
from argparse import ArgumentParser


def log(msg, *args):
    if len(args) > 0:
        msg = str(msg) % args
    print msg

def _convert_from_lossy_integers(X, precision=1000, dtype=np.float32):
    """Implementation of the lossy compression used in Gromacs XTC using
    the pytables library.  Convert 16 bit integers into 32 bit floats."""
    X2 = X.astype(dtype)
    X2 /= float(precision)
    return X2

    
parser = ArgumentParser('k-means RMSD clustering')
parser.add_argument('-a', '--atom_indices', required=True)
parser.add_argument('-p', '--project_yaml', required=True)
parser.add_argument('-k', '--n_clusters', required=True, type=int)
parser.add_argument('-e', '--epsilon', help='convergence criterion', default=1e-8, type=float)
parser.add_argument('-t', '--max_time', help='max execution time (seconds)', default=None, type=int)


args = parser.parse_args()
log('args: %s', args)

log('opening project file...')
with open(args.project_yaml) as f:
    project = yaml.load(f)

atom_indices = np.loadtxt(args.atom_indices, int)
n_frames = sum(t['length'] for t in project['trajs'])
n_atoms = len(atom_indices)
n_padded_atoms = 4 + n_atoms - n_atoms % 4

log('allocating array of size %.1f MB...', n_frames*3*n_padded_atoms*8 / (1024.0**2))
# we're asking for float64 here because currently the kmeans code require double
# (it will make a single precision copy for the RMSD code itself)
xyzlist = np.zeros((n_frames, 3, n_padded_atoms), dtype=np.float64)
xyzlist_pointer = 0

log('loading trajectories...')
for traj_record in project['trajs']:
    path = os.path.join(os.path.dirname(args.project_yaml), traj_record['path'])
    with tables.File(path) as f:
        # xyz contains the trajectory data from this file
        # we reshape it into axis major ordering
        xyz = f.root.XYZList[:, atom_indices, :].swapaxes(1,2)
    assert xyz.shape[1] == 3, 'not in axis major ordering'
    if xyz.dtype == np.int16:
        xyz = _convert_from_lossy_integers(xyz, dtype=np.float64)

    # and accumulate the data into xyzlist
    xyzlist[xyzlist_pointer:xyzlist_pointer+len(xyz), :, 0:n_atoms] = xyz
    xyzlist_pointer += len(xyz)


log('clustering...')
kmeans_mds(xyzlist, k=args.n_clusters, threshold=args.epsilon, max_time=args.max_time)
