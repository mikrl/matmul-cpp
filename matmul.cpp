#include<vector>

#include "matrix_utils.h"

std::vector<std::vector<float>> matmul(std::vector<std::vector<float>> A, std::vector<std::vector<float>> B){
    // Cij = 
    int A_rows = count_rows(A);
    int A_cols = count_cols(A);
    
    int B_rows = count_rows(B);
    int B_cols = count_cols(B);

    if (A_cols != B_rows){
        throw;
    }

    int C_rows = A_cols; 
    int C_cols = B_rows;

    std::vector<std::vector<float>> C;

    for (int col_idx = 0; col_idx < C_rows; col_idx++){
        std::vector<float> row(C_rows);          
        C.push_back(row);
    }

    return C;
}