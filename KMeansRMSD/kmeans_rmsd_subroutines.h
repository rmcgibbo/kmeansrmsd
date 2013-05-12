// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef KMEANS_RMSD_H
#define KMEANS_RMSD_H

int gower_matrix(double* X, int X_dim0, int X_dim1, int X_dim2, int X_dim2_mem,
    long* assignments, int assignments_dim0, long k,
    double* B, int B_dim0, int B_dim1);
    
int average_structure(double* X, int X_dim0, int X_dim1, int X_dim2, int X_dim2_mem,
                      long* assignments, int assignments_dim0, long k,
                      double* R, int R_dim0, int R_dim1, int R_dim1_mem);

int is_mirror_image(double* X, int X_dim0, int X_dim1, int X_dim1_mem,
                    double* Y, int Y_dim0, int Y_dim1, int Y_dim1_mem);
                          
int rectify_mirror(double* X, int X_dim0, int X_dim1, int X_dim1_mem,
                   double* Y, int Y_dim0, int Y_dim1, int Y_dim1_mem);
    
int center_inplace(double* X, int X_dim0, int X_dim1, int X_dim1_mem);

#endif /*KMEANS_RMSD_H*/