int printout();

int gower_matrix(double* X, int X_dim0, int X_dim1, int X_dim2,
    long* assignments, int assignments_dim0, long k,
    double* B, int B_dim0, int B_dim1);
    
int average_structure(double* X, int X_dim0, int X_dim1, int X_dim2,
                      long* assignments, int assignments_dim0, long k,
                      double* R, int R_dim0, int R_dim1);
                      
int is_mirror_image(double* X, int X_dim0, int X_dim1, double* Y, int Y_dim0, int Y_dim1);

int center_inplace(double* X, int X_dim0, int X_dim1);

int rectify_mirror(double* X, int X_dim0, int X_dim1, double* Y, int Y_dim0, int Y_dim1);