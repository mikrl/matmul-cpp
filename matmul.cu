#define ARRAYSIZE(arr) (sizeof(arr) / sizeof(arr[0]))

#include<vector>
#include "matrix_utils.h"

__global__ void matmul_GPU(float** A, float** B, float** C){
    /* Simple matmul routine for a single GPU core.
       Assumes that input tensors have the correct dimensions for a matrix multiplication.
       Assumes output tensor has the correct dimension for a matrix multiplication. 
    */ 
    int i = threadIdx.x;
    int j = threadIdx.y;

    // int B_rows = count_rows(B); 
    int B_rows = ARRAYSIZE(B);

    float subsum = 0;
    for(int k=0; k<B_rows; k++) subsum+=A[i][k]*B[k][j];
    C[i][j] = subsum;
}

std::vector<std::vector<float>> matmul_cuda(std::vector<std::vector<float>> A, std::vector<std::vector<float>> B){
   
    int A_cols = A[0].size();
    int B_rows = B.size();
    
    int num_blocks = 1;
    dim3 threads_per_block(A_cols, B_rows);

    float** A_arr = matrix_to_arr(A);
    float** B_arr = matrix_to_arr(B);

    float** C_arr;

    matmul_GPU<<<num_blocks, threads_per_block>>>(A_arr, B_arr, C_arr);

    std::vector<std::vector<float>> C = arr_to_matrix(C_arr, A_cols, B_rows);

    return C;

}

