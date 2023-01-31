#include<vector>

#include "matrix_utils.h"



std::vector<std::vector<float>> matmul_simple(std::vector<std::vector<float>> A, std::vector<std::vector<float>> B){

    check_compatibility(A, B);
    
    int A_cols = count_cols(A);
    int B_rows = count_rows(B);
    
    auto C = allocate_empty_matrix(A_cols, B_rows);

    for(int i=0; i<A_cols; i++){
        for(int j=0; j<B_rows; j++){
            float subsum = 0;
            for(int k=0; k<B_rows; k++) subsum+=A[i][k]*B[k][j];
            C[i][j] = subsum;
        }
    }   
    return C;
}