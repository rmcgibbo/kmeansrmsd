#include <iostream>
#include <vector>
#include <stdlib.h>
#include <string>
#include <assert.h>
#include <math.h>

#include "lib.h"
#include "cblas.h"
#include "clapack.h"

using namespace std;

int printout() {
    cout << "std";
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
    // X : double*
    //     pointer to the upper left corner of the conformation's coordinares.
    // X_dim0 : int
    //     number of rows of X. This should be the number of atoms.
    // X_dim1 : int
    //     number of columns of X. This should be 3.
    
    if (X_dim1 != 3) {
        cout << "X_dim1 is not 3";
        exit(1);
    }
    int i;
    double mu_X = 0;
    double mu_Y = 0;
    double mu_Z = 0;

    for (i = 0; i < X_dim0; i++) {
        mu_X += X[i*3 + 0] / X_dim0;
        mu_Y += X[i*3 + 1] / X_dim0;
        mu_Z += X[i*3 + 2] / X_dim0;
    }
    
    for (i = 0; i < X_dim0; i++) {
        X[i*3 + 0] -= mu_X;
        X[i*3 + 1] -= mu_Y;
        X[i*3 + 2] -= mu_Z;
    }
}

int rectify_mirror(double* X, int X_dim0, int X_dim1, double* Y, int Y_dim0, int Y_dim1) {
    if ((X_dim1 != 3) || (Y_dim1 != 3)) {
        exit(1);
    }
    
    center_inplace(X, X_dim0, X_dim1);
    center_inplace(X, X_dim0, Y_dim1);

    int i, j;
    i = 0;
    while ((i < 2) && is_mirror_image(X, X_dim0, X_dim1, Y, Y_dim0, Y_dim1)) {
        for (j = 0; j < X_dim0; j++) {
            X[j*X_dim1 + i] *= -1;
        }
        i++;
    }
    
    if (is_mirror_image(X, X_dim0, X_dim1, Y, Y_dim0, Y_dim1)) {
        cout << "It should NOT be a mirror image now";
        exit(1);
    }
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
    
    if ((X_dim0 != Y_dim0) || (X_dim1 != Y_dim1) || (X_dim1 != 3)){
        cout << "is_mirror_image called with wrong shape" << endl;
        exit(1);
    }
    
    // center_inplace(X, X_dim0, X_dim1);
    // center_inplace(Y, Y_dim0, Y_dim1);

    // covariance = dot(X.T, Y)
    double covariance[9];
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 3, 3, X_dim0, 1.0, X, 3, Y, 3, 0.0, covariance, 3);

    // cout << "(C) Covariance" << endl;
    // for (i = 0; i < 3; i++) {
    //     for (j = 0; j < 3; j++) {
    //         cout << covariance[i*3+j] << ",  ";
    //     }
    //     cout << endl;
    // }
    
    // Now take the SVD of the covariance matrix
    int three = 3;
    double singular_values[3];
    double U[9];
    double VT[9];
    // the workspace is of size 201 because that's what the mac
    // lapack said was optimal for 3x3
    int lwork = 201;
    double work[lwork];
    int info = 0; 
    
    dgesvd_((char *)"A", (char *)"A", &three,  &three, covariance, &three,
        singular_values, U, &three, VT, &three, work, &lwork, &info);
    if (info != 0) {
        cout << "dgesvd_ INFO code: " << info << endl;
        cout << "< 0:  if INFO = -i, the i-th argument had an illegal value." << endl;
        cout << "> 0:  if DBDSQR did not converge, INFO specifies how many" << endl;
        cout << "       superdiagonals of an intermediate bidiagonal form B" << endl;
        cout << "       did not converge to zero. See the description of WORK" << endl;
        cout << "       above for details" << endl;
        exit(1);
    }
    
    // this is not the determinant of the rotation matrix
    double determinant = ddet(U) * ddet(VT);
    // cout << "(C) Determinant" << endl;
    // cout << determinant << endl;
    return determinant < 0;
}


int gower_matrix(double* X, int X_dim0, int X_dim1, int X_dim2,
    long* assignments, int assignments_dim0, long k,
    double* B, int B_dim0, int B_dim1) {
    // Gower, J.C. (1966). Some distance properties of latent root
    // and vector methods used in multivariate analysis.
    // Biometrika 53: 325-338
    
    // Parameters
    // ----------
    // X : array, shape=(n_frames, n_atoms, 3)
    // frame_indices, shape=(n_frame_indices)
    
    assert (X_dim2 == 3);
    assert (X_dim1 == B_dim0);
    assert (X_dim1 == B_dim2);
    assert (assignments_dim0 == X_dim0);
    
    int retval = -1;
    int n_assignments = 0;
    long i, j;
    
    // count the number of assignments
    for (i = 0; i < assignments_dim0; i++) {
        if (assignments[i] == k) {
            retval = i;
            n_assignments++;
        }
    }
    if (n_assignments <= 0) {
        // cout << "gower_matrix() assignments" << endl;
        // for (i = 0; i < assignments_dim0; i++) {
        //     cout << assignments[i] << ", ";
        // }
        // cout << endl << "No assignments for k= " << k;
        return -1;
    }

    for (i = 0; i < assignments_dim0; i++) {
        if (assignments[i] != k) {
            continue;
        }
        // B += (1/n_frames) X[i] * X[i].T
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,    // order, trans, transb
                    X_dim1, X_dim1, X_dim2,          // M, N, K
                    1.0 / n_assignments,             // alpha
                    &X[i*X_dim1*X_dim2], X_dim2,     // A, lda
                    &X[i*X_dim1*X_dim2], X_dim2,     // B, ldb
                    1.0,  B, X_dim1);                // beta, C, ldc
    }
    
    double ones[X_dim1];
    double b[X_dim1];
    std::fill_n(ones, X_dim1, 1.0);
    
    // mean the matrix B along axis 1, putting the results in b
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                X_dim1,  X_dim1, 1.0 / X_dim1,
                B, X_dim1, ones, 1, 0.0, b,1);
    
    // mean the vector b
    double bb = cblas_ddot(X_dim1, b, 1, ones, 1) / X_dim1;    

    for (int i = 0; i < X_dim1; i++) {
        for (int j = 0; j < X_dim1; j++) {
            B[i*X_dim1 + j] +=  bb - (b[i] + b[j]);
        }
    }
    
    return retval;
}


int average_structure(double* X, int X_dim0, int X_dim1, int X_dim2,
                      long* assignments, int assignments_dim0, long k,
                      double* R, int R_dim0, int R_dim1) {
    if ((X_dim1 != R_dim0) || (X_dim2 != R_dim1) || (X_dim2 != 3)){
        cout << "average_structure called with wrong shape" << endl;
        exit(1);
    }
    
    int status = 0;
    // declare the workspace for the gower matrix
    double B[X_dim1*X_dim1];
    memset(B, 0, sizeof(double)*X_dim1*X_dim1);
    
    status = gower_matrix(X, X_dim0, X_dim1, X_dim2, assignments, assignments_dim0, k,
                          B, X_dim1, X_dim1);
    
    if (status == -1) {
        int new_seed = rand() % X_dim0;
        cout << "Warning: No assignments for state " << k << ". ";
        cout << "Choosing new seed structure: " << new_seed << endl;        
        memcpy(R, &X[new_seed*X_dim1*X_dim2], X_dim1*X_dim2*sizeof(double));
        return 0;
    }
    
    
    // Query the size of the workspace that the eigensolver wants
    // this is done by setting lwork=-1, and the size gets put in work[0]
    int info;
    int lwork = -1;
    double work[1];
    dsyev_((char *)"V", (char *)"U", &X_dim1, NULL, &X_dim1, NULL, work, &lwork, &info);
    // cout << "WORK SPACE " << work[0] << endl;
    lwork = work[0];

    // call the eigensolver. this puts the eigenvectors into B, and the
    // eigenvalus into `eigenvalues`
    double work2[lwork];
    double eigenvalues[X_dim1];
    dsyev_((char *)"V", (char *)"U", &X_dim1, B, &X_dim1, eigenvalues, work2, &lwork, &info);
    if (info != 0) {
        cout << "LAPACK dsyev INFO code: " << info << endl;
        cout << "    < 0:  if INFO = -i, the i-th argument had an illegal value" << endl;
        cout << "    > 0:  if INFO = i, the algorithm failed to converge; i" << endl;
        cout << "          off-diagonal elements of an intermediate tridiagonal" << endl;
        cout << "          form did not converge to zero." << endl;
        exit(1);
    }
    
    int i, j, column;
    for (i = 0; i < X_dim1; i++ ) {
        for (j = 0; j < 3; j++) {
            column = j + X_dim1 - 3;
            R[i*3 + j] = B[column*X_dim1 + i] * sqrt(eigenvalues[column]);
        }
    }
    
    rectify_mirror(R, R_dim0, R_dim1, &X[status*X_dim1*X_dim2], X_dim1, X_dim2);
    
    return 1;
}

