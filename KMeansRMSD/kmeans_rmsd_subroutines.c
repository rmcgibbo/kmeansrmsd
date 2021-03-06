
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "kmeans_rmsd_subroutines.h"

#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>


int gsl_matrix_printf(gsl_matrix *m) {
    // Print a gsl_matrix to stdout
    
    size_t rows=m->size1;
    size_t cols=m->size2;
    size_t row,col,ml;
    int fill;
    char buf[100];
    gsl_vector *maxlen;

    maxlen=gsl_vector_alloc(cols);
    for (col=0;col<cols;++col) {
            ml=0;
            for (row=0;row<rows;++row) {
                    sprintf(buf, "%f", gsl_matrix_get(m,row,col));
                    if (strlen(buf)>ml)
                            ml=strlen(buf);
            }
            gsl_vector_set(maxlen,col,ml);
    }

    for (row=0;row<rows;++row) {
            for (col=0;col<cols;++col) {
                    sprintf(buf, "%f", gsl_matrix_get(m,row,col));
                    fprintf(stdout,"%s",buf);
                    fill=gsl_vector_get(maxlen,col)+2-strlen(buf);
                    while (--fill>=0)
                            fprintf(stdout," ");
            }
            fprintf(stdout,"\n");
    }
    gsl_vector_free(maxlen);
    return 0;
}


inline double ddet(double* A) {
    // Compute the determinant of a 3x3 matrix
    
    // Parameters
    // ----------
    // A : double*, shape=(3,3)
    //     pointer to the uppler left corner of the 3x3 matrix A

    // Returns
    // -------
    // determinant : double
    //     the determinant, computed with the rule of saurrus.
    
    return A[0]*A[4]*A[8] + A[1]*A[5]*A[6] + A[2]*A[3]*A[7]
         - A[2]*A[4]*A[6] - A[5]*A[7]*A[0] - A[8]*A[1]*A[3];
}

int center_inplace(double* X, int X_dim0, int X_dim1, int X_dim1_mem) {
    // Center a conformation in place
    // 
    // Parameters
    // ----------
    // X : double*, shape=(X_dim0, X_dim1)
    //     pointer to the upper left corner of the conformation's coordinares.
    // X_dim0 : int
    //     number of rows of X. Should be 3
    // X_dim1 : int
    //     number of columns of X. Corresponds to the number of atoms
    // X_dim1_mem : int
    //     number of columns of X in memory. corresponds to the number of padded atoms.
    //     such that the (i,j)-th element of X is accessed at X[i*X_dim1*mem + j]
    
    if (X_dim0 != 3) {
        fprintf(stderr, "X_dim0 is not 3\n");
        exit(1);
    }
    int i, j;

    for (j = 0; j < X_dim0; j++) {
        double mu  = 0;
        for (i = 0; i < X_dim1; i++) {
            mu += X[j*X_dim1_mem + i] / X_dim1;
        }

        for (i = 0; i < X_dim1; i++) {
            X[j*X_dim1_mem + i] -= mu;
        }
    }
    return 0;
}

int rectify_mirror(double* X, int X_dim0, int X_dim1, int X_dim1_mem,
                   double* Y, int Y_dim0, int Y_dim1, int Y_dim1_mem) {
    // Swap the direction of the axes in a conformation so that its optimal
    // alignment with respect to a second frame involves no mirror inversion
    // 
    // This routine modifies the conformation X inplace, by reversing the
    // direction of one or more of its cartesian directions -- i.e sending
    // x to -x, y to -y and or z to -z such thats its optimal alignment with
    // respect to a secon frame (Y), involves pure rotation and no inversion.
    // 
    //  X will also be centered. Conformation Y *SHOULD ALREADY BE CENTERED*
    // 
    // Parameters
    // ----------
    // X : double*, shape=(X_dim0, X_dim1)
    //    Pointer to the upper left corner of matrix X. This is the conformation
    //    that will be modified
    // X_dim0 : int
    //    The number of rows in matrix X. Should be 3.
    // X_dim1 : int
    //    The number of columns in matrix X. Corresponds to the number of atoms
    // X_dim1_mem : int
    //     number of columns of X in memory. corresponds to the number of padded atoms.
    //     such that the (i,j)-th element of X is accessed at X[i*X_dim1*mem + j]
    // Y : double*, shape=(X_dim0, X_dim1)
    //    Pointer to the upper left corner of matrix X. This is the "reference"
    //    conformation.
    // Y_dim0 : int
    //    The number of rows in matrix Y. Should be 3.
    // Y_dim1 : int
    //    The number of columns in matrix Y. Corresponds to the number of atoms.
    // Y_dim1_mem : int
    //     number of columns of Y in memory. corresponds to the number of padded atoms.
    //     such that the (i,j)-th element of Y is accessed at Y[i*Y_dim1*mem + j]

    if ((X_dim0 != 3) || (Y_dim0 != 3)) {
        fprintf(stderr, "rectify_mirror called with incorrect shape\n");
        exit(1);
    }
    
    center_inplace(X, X_dim0, X_dim1, X_dim1_mem);
    //center_inplace(Y, Y_dim0, Y_dim1, Y_dim1_mem);

    int i, j;
    i = 0;
    while ((i < 2) && is_mirror_image(X, X_dim0, X_dim1, X_dim1_mem, Y, Y_dim0, Y_dim1, Y_dim1_mem)) {
        for (j = 0; j < X_dim1; j++) {
            X[i*X_dim1 + j] *= -1;
        }
        i++;
    }
    
    if (is_mirror_image(X, X_dim0, X_dim1, X_dim1_mem, Y, Y_dim0, Y_dim1, Y_dim1_mem)) {
        fprintf(stderr, "It should NOT be a mirror image now\n");
        exit(1);
    }
    
    return 0;
}


int is_mirror_image(double* X, int X_dim0, int X_dim1, int X_dim1_mem,
                    double* Y, int Y_dim0, int Y_dim1, int Y_dim1_mem) {
    // Check if two configurations X and Y are mirror images
    // (i.e. does their optimal superposition involve a reflection?)
    // 
    // Parameters
    // ----------
    // X : double*, shape=(X_dim0, X_dim1)
    //    Pointer to the upper left corner of matrix X.
    // X_dim0 : int
    //    The number of rows in matrix X. Should be 3.
    // X_dim1 : int
    //    The number of columns in matrix X. Corresponds to number of atoms
    // X_dim1_mem : int
    //     number of columns of X in memory. corresponds to the number of padded atoms.
    //     such that the (i,j)-th element of X is accessed at X[i*X_dim1*mem + j]
    // Y : double*, shape=(X_dim0, X_dim1)
    //    Pointer to the upper left corner of matrix X.
    // Y_dim0 : int
    //    The number of rows in matrix Y. Should be 3.
    // Y_dim1 : int
    //    The number of columns in matrix Y. Corresponds to number of atoms
    // Y_dim1_mem : int
    //     number of columns of Y in memory. corresponds to the number of padded atoms.
    //     such that the (i,j)-th element of Y is accessed at Y[i*Y_dim1*mem + j]    // 
    //
    // Returns
    // -------
    // mirror : int
    //     = 1 if they are mirror images
    //     = 0 if they are not mirror images
    
    if ((X_dim0 != Y_dim0) || (X_dim1 != Y_dim1) || (X_dim0 != 3)){
        fprintf(stderr, "is_mirror_image called with wrong shape\n");
        exit(1);
    }
    
    // covariance = np.dot(X, Y.T)
    gsl_matrix* covariance = gsl_matrix_alloc(3, 3);
    gsl_matrix_view mX = gsl_matrix_view_array_with_tda(X, X_dim0, X_dim1, X_dim1_mem);
    gsl_matrix_view mY = gsl_matrix_view_array_with_tda(Y, Y_dim0, Y_dim1, Y_dim1_mem);
    
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, &mX.matrix, &mY.matrix, 0.0, covariance);
    
    gsl_matrix* U = gsl_matrix_alloc(3, 3);
    gsl_vector* S = gsl_vector_alloc(3);
    gsl_vector* work = gsl_vector_alloc(3);
    
    gsl_linalg_SV_decomp(covariance, U, S, work);
    double determinant = ddet(covariance->data) * ddet(U->data);
    
    gsl_matrix_free(covariance);
    gsl_matrix_free(U);
    gsl_vector_free(S);
    gsl_vector_free(work);
    
    return determinant < 0;
}


int gower_matrix(double* X, int X_dim0, int X_dim1, int X_dim2, int X_dim2_mem,
                 long* assignments, int assignments_dim0, long k,
                 double* B, int B_dim0, int B_dim1) {
    // Compute the Gower matrix over an ensemble of conformations.
    // 
    // The result can be thought of as the average dissimilarity between each
    // of the atoms.
    //
    // Gower, J.C. (1966). Some distance properties of latent root
    // and vector methods used in multivariate analysis.
    // Biometrika 53: 325-338
    // 
    // Parameters
    // ----------
    // X : double*, shape=(X_dim0, X_dim1, X_dim2)
    //     The centered cartesian coordinates
    // X_dim0 : int
    //     number of rows of X. Corresponds to the number of frames.
    // X_dim1 : int
    //     number of columns of X. This should be 3.
    // X_dim2_mem : int
    //     If the array on disk has "padded" atoms, then X_dim2_mem should be
    //     the number of atoms with padding. This is important because we
    //     need to skip over the right number of frames to find the n-th
    //     conformation on disk.
    // assignments : long*, shape=(assignments_dim0)
    //     The assignments for each frame.
    // k : long
    //     The requested state. Only frames with index i such that assignments[i] == k
    //     will be averaged.
    // B : double*, shape=(B_dim0, B_dim1)
    //     The output array. It's a dissimilarity matrix over the atoms, so
    //     the dimenensions should be n_atoms x n_atoms (that is, X_dim2 x X_dim2)
    // 
    // Returns
    // -------
    // status : int
    //     = -1    if none of the frames have assignment[k]. this is an error.
    //     > 0     when the method runs correctly, the return code will contain
    //             the index of the last frame assigned to state k.
    
    if ((X_dim1 != 3) || (X_dim2 != B_dim0) || (X_dim2 != B_dim1) || (assignments_dim0 != X_dim0)){
        fprintf(stderr, "gower_matrix called with wrong shapes\n");
        exit(1);
    }
    
    gsl_matrix_view mA, mB;
    int retval = -1;
    int n_assignments = 0;
    int i, j;
    long p;
    
    // count the number of assignments
    for (i = 0; i < assignments_dim0; i++) {
        if (assignments[i] == k) {
            retval = i;
            n_assignments++;
        }
    }
    if (n_assignments <= 0) {
        fprintf(stderr, "no assignements (gower)\n");
        return -1;
    }
    mB = gsl_matrix_view_array(B, B_dim0, B_dim1);

    for (p = 0; p < assignments_dim0; p++) {
        if (assignments[p] != k) {
            continue;
        }
        
        mA = gsl_matrix_view_array_with_tda(&X[p*X_dim1*X_dim2_mem], X_dim1, X_dim2, X_dim2_mem);
        // for (j = 0; j < X_dim2; j++) {
            // printf("%f", gsl_matrix_get(&mA.matrix, 0, j));
        // }
        
        // B += (1/n_frames) X.T[i] * X[i]
        gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0 / n_assignments,
            &mA.matrix, &mA.matrix, 1.0, &mB.matrix);
    }
     
    // ones = np.ones(X_dim2)
    // b = np.empty(X_dim2)
    gsl_vector* ones = gsl_vector_alloc(X_dim2);
    gsl_vector_set_all(ones, 1.0);
    gsl_vector* b = gsl_vector_alloc(X_dim2);
    
    // b = np.mean(B) / B.shape[0]
    gsl_blas_dgemv(CblasNoTrans, 1.0 / X_dim2, &mB.matrix, ones, 0.0, b);
    
    // bb = np.mean(b)
    double bb;
    gsl_blas_ddot(b, ones, &bb);
    bb /= X_dim2;
    
    // B = B - np.add.outer(b, b)) + bb
    for (i = 0; i < X_dim2; i++) {
        for (j = 0; j < X_dim2; j++) {
            B[i*X_dim2 + j] +=  bb - (gsl_vector_get(b, i) + gsl_vector_get(b, j));
        }
    }
    
    gsl_vector_free(ones);
    gsl_vector_free(b);
    return retval;
}


int average_structure(double* X, int X_dim0, int X_dim1, int X_dim2, int X_dim2_mem,
                      long* assignments, int assignments_dim0, long k,
                      double* R, int R_dim0, int R_dim1, int R_dim1_mem) {
    // Compute an "average conformation" from amongst the conformations
    // in xyzlist[assignments==k]
    // 
    // Parameters (input)
    // ------------------
    // X : double*
    //     pointer to the upper left corner of the trajectoy's coordinates.
    //     X should be the start of a 3d matrix.
    // X_dim0 : int
    //     number of rows of X. Corresponds to the number of frames.
    // X_dim1 : int
    //     number of columns of X. This should be 3.
    // X_dim2 : int
    //     size of the third dimension of X. Corresponds to the number of atoms.
    // X_dim2_mem : int
    //     If the array on disk has "padded" atoms, then X_dim2_mem should be
    //     the number of atoms with padding. This is important because we
    //     need to skip over the right number of frames to find the n-th
    //     conformation on disk.
    // assignments : long*
    //     pointer to the beginning of the assignments vector, which contains
    //     the index of the "state" that each conformation is assigned to.
    // k : long
    //     this routine will only touch entries in X corresponding to frames
    //     whose assignment is equal to k. The other frames will be skipped.
    // 
    // Parameters (output)
    // -------------------
    // R : double*
    //     pointer to the start of a conformation where you'd like the resulting
    //     average structure stored.
    // R_dim0 : int
    //     number of rows of R. this should be 3
    // R_dim1 : int
    //     number of columns of R. corresponds to the number of atoms
    
    if ((X_dim1 != R_dim0) || (X_dim2 != R_dim1) || (X_dim1 != 3)){
        fprintf(stderr, "X_dim1 %d\n", X_dim1);
        fprintf(stderr, "R_dim0 %d\n", R_dim0);
        fprintf(stderr, "X_dim2 %d\n", X_dim2);   
        fprintf(stderr, "R_dim1 %d\n", R_dim1);
        fprintf(stderr, "average_structure called with wrong shape\n");
        exit(1);
    }
    if (X_dim2_mem <= X_dim2) {
        fprintf(stderr, "x_dim2_mem must be greater than or equal to X_dim2");
        exit(1);
    }
    
    int status = 0;
    // declare the workspace for the gower matrix
    double B[X_dim2*X_dim2];
    memset(B, 0, sizeof(double)*X_dim2*X_dim2);
    
    status = gower_matrix(X, X_dim0, X_dim1, X_dim2, X_dim2_mem, assignments, assignments_dim0, k,
                          B, X_dim2, X_dim2);
    
    if (status == -1) {
        int new_seed = rand() % X_dim0;
        fprintf(stderr, "Warning: No assignments for state %ld\n", k);
        fprintf(stderr, "Choosing new seed structure: %d\n", new_seed);
        memcpy(R, &X[new_seed*X_dim1*X_dim2_mem], X_dim2*sizeof(double));
        memcpy(R + X_dim2_mem, &X[new_seed*X_dim1*X_dim2_mem + X_dim2_mem], X_dim2*sizeof(double));
        memcpy(R + 2*X_dim2_mem, &X[new_seed*X_dim1*X_dim2_mem + 2*X_dim2_mem], X_dim2*sizeof(double));
        return 0;
    }

    gsl_matrix_view mB = gsl_matrix_view_array(B, X_dim2, X_dim2);
    gsl_eigen_symmv_workspace* workspace = gsl_eigen_symmv_alloc(X_dim2);
    gsl_vector* eval = gsl_vector_alloc(X_dim2);
    gsl_matrix* evec = gsl_matrix_alloc(X_dim2, X_dim2);

    gsl_eigen_symmv(&mB.matrix, eval, evec, workspace);
    gsl_eigen_symmv_free(workspace);
    gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_VAL_DESC);
    
    // printf("Eigenvectors\n");
    // gsl_matrix_printf(evec);
    // printf("\n");
    
    int i;
    gsl_vector_view column;
    for (i = 0; i < X_dim2; i++) {
        column = gsl_matrix_column(evec, i);
        gsl_vector_scale(&column.vector, sqrt(gsl_vector_get(eval, i)));
    }
    
    gsl_matrix_view output = gsl_matrix_view_array_with_tda(R, R_dim0, R_dim1, R_dim1_mem);
    gsl_matrix_view submatrix = gsl_matrix_submatrix(evec, 0, 0, X_dim2, 3);
    gsl_matrix_transpose_memcpy(&output.matrix, &submatrix.matrix);

    rectify_mirror(R, R_dim0, R_dim1, R_dim1_mem, &X[status*X_dim1*X_dim2_mem], X_dim1, X_dim2, X_dim2_mem);

    gsl_vector_free(eval);
    gsl_matrix_free(evec);
    return 1;
}

