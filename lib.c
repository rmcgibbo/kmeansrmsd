// #include <iostream>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "lib.h"
// #include "cblas.h"
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
// #include "clapack.h"

// using namespace std;

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
    // A : double*
    //     pointer to the uppler left corner of the 3x3 matrix A

    // Returns
    // -------
    // determinant : double
    //     the determinant, computed with the rule of saurrus.
    
    return A[0]*A[4]*A[8] + A[1]*A[5]*A[6] + A[2]*A[3]*A[7]
         - A[2]*A[4]*A[6] - A[5]*A[7]*A[0] - A[8]*A[1]*A[3];
}

int center_inplace(double* X, int X_dim0, int X_dim1) {
    // Center a conformation in place
    // 
    // Parameters
    // ----------
    // X : double*
    //     pointer to the upper left corner of the conformation's coordinares.
    // X_dim0 : int
    //     number of rows of X. This should be the number of atoms.
    // X_dim1 : int
    //     number of columns of X. This should be 3.
    
    if (X_dim0 != 3) {
        fprintf(stderr, "X_dim0 is not 3\n");
        exit(1);
    }
    int i, j;

    for (j = 0; j < X_dim0; j++) {
        double mu  = 0;
        for (i = 0; i < X_dim1; i++) {
            mu += X[j*X_dim1 + i] / X_dim1;
        }

        for (i = 0; i < X_dim1; i++) {
            X[j*X_dim1 + i] -= mu;
        }
    }
    return 0;
}

int rectify_mirror(double* X, int X_dim0, int X_dim1,
                   double* Y, int Y_dim0, int Y_dim1) {
    // Swap the direction of the axes in a conformation so that its optimal
    // alignment with respect to a second frame involves no mirror inversion
    // 
    // This routine modifies the conformation X inplace, by reversing the
    // direction of one or more of its cartesian directions -- i.e sending
    // x to -x, y to -y and or z to -z such thats its optimal alignment with
    // respect to a secon frame (Y), involves pure rotation and no inversion.
    // 
    // Parameters
    // ----------
    // X : double*, shape=(X_dim0, X_dim1)
    //    Pointer to the upper left corner of matrix X. This is the conformation
    //    that will be modified
    // X_dim0 : int
    //    The number of rows in matrix X. This should be the number of atoms.
    // X_dim1 : int
    //    The number of columns in matrix X. This should be equal to 3.
    // Y : double*, shape=(X_dim0, X_dim1)
    //    Pointer to the upper left corner of matrix X. This is the "reference"
    //    conformation.
    // Y_dim0 : int
    //    The number of rows in matrix Y. This should be the number of atoms.
    // Y_dim1 : int
    //    The number of columns in matrix Y. This should be equal to 3.

    if ((X_dim0 != 3) || (Y_dim0 != 3)) {
        fprintf(stderr, "rectify_mirror called with incorrect shape\n");
        exit(1);
    }
    
    center_inplace(X, X_dim0, X_dim1);
    center_inplace(Y, Y_dim0, Y_dim1);

    int i, j;
    i = 0;
    while ((i < 2) && is_mirror_image(X, X_dim0, X_dim1, Y, Y_dim0, Y_dim1)) {
        for (j = 0; j < X_dim1; j++) {
            X[i*X_dim1 + j] *= -1;
        }
        i++;
    }
    
    if (is_mirror_image(X, X_dim0, X_dim1, Y, Y_dim0, Y_dim1)) {
        fprintf(stderr, "It should NOT be a mirror image now\n");
        exit(1);
    }
    
    return 0;
}


int is_mirror_image(double* X, int X_dim0, int X_dim1, double* Y, int Y_dim0, int Y_dim1) {
    // Check if two configurations X and Y are mirror images
    // (i.e. does their optimal superposition involve a reflection?)
    // 
    // Parameters
    // ----------
    // X : double*, shape=(X_dim0, X_dim1)
    //    Pointer to the upper left corner of matrix X.
    // X_dim0 : int
    //    The number of rows in matrix X. This should be the number of atoms.
    // X_dim1 : int
    //    The number of columns in matrix X. This should be equal to 3.
    // Y : double*, shape=(X_dim0, X_dim1)
    //    Pointer to the upper left corner of matrix X.
    // Y_dim0 : int
    //    The number of rows in matrix Y. This should be the number of atoms.
    // Y_dim1 : int
    //    The number of columns in matrix Y. This should be equal to 3.
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
    gsl_matrix_view mX = gsl_matrix_view_array(X, X_dim0, X_dim1);
    gsl_matrix_view mY = gsl_matrix_view_array(Y, X_dim0, X_dim1);
    
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


int gower_matrix(double* X, int X_dim0, int X_dim1, int X_dim2,
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
    
    // Parameters
    // ----------
    // X : array, shape=(n_frames, n_atoms, 3)
    // frame_indices, shape=(n_frame_indices)
    
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
        return -1;
    }
    mB = gsl_matrix_view_array(B, B_dim0, B_dim1);

    for (p = 0; p < assignments_dim0; p++) {
        if (assignments[p] != k) {
            continue;
        }
        
        mA = gsl_matrix_view_array(&X[p*X_dim1*X_dim2], X_dim1, X_dim2);
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


int average_structure(double* X, int X_dim0, int X_dim1, int X_dim2,
                      long* assignments, int assignments_dim0, long k,
                      double* R, int R_dim0, int R_dim1) {
    if ((X_dim1 != R_dim0) || (X_dim2 != R_dim1) || (X_dim1 != 3)){
        fprintf(stderr, "average_structure called with wrong shape\n");
        exit(1);
    }
    
    int status = 0;
    // declare the workspace for the gower matrix
    double B[X_dim2*X_dim2];
    memset(B, 0, sizeof(double)*X_dim2*X_dim2);
    
    status = gower_matrix(X, X_dim0, X_dim1, X_dim2, assignments, assignments_dim0, k,
                          B, X_dim2, X_dim2);
    
    if (status == -1) {
        int new_seed = rand() % X_dim0;
        fprintf(stderr, "Warning: No assignments for state %ld\n", k);
        fprintf(stderr, "Choosing new seed structure: %d\n", new_seed);
        memcpy(R, &X[new_seed*X_dim1*X_dim2], X_dim1*X_dim2*sizeof(double));
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
    
    gsl_matrix_view output = gsl_matrix_view_array(R, R_dim0, R_dim1);
    gsl_matrix_view submatrix = gsl_matrix_submatrix(evec, 0, 0, X_dim2, 3);
    gsl_matrix_transpose_memcpy(&output.matrix, &submatrix.matrix);

    rectify_mirror(R, R_dim0, R_dim1, &X[status*X_dim1*X_dim2], X_dim1, X_dim2);

    gsl_vector_free(eval);
    gsl_matrix_free(evec);
    return 1;
}

