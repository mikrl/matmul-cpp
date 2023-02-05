#include<vector>
#include "matrix_utils.h"


struct ctlInput{
    int inner_product_size; // The size of A_rows or B_cols to take the sum sum over
    int out_rows;
    int out_cols;
};

__global__ void matmul_GPU(ctlInput *control, float *A_arr, float *B_arr, float *output){
    /* Simple matmul routine for a single GPU core.
       Assumes that input tensors have the correct dimensions for a matrix multiplication.
       Assumes output tensor has the correct dimension for a matrix multiplication. 
    */ 

    int i = threadIdx.x;
    int j = threadIdx.y;

    int inner_product_size = control->inner_product_size;
    int output_dims[2] = {control->out_rows, control->out_cols};

    float subsum = 0;
    for(int k=0; k<inner_product_size; k++){
        float A_row = get_linear_index(output_dims, i, k);
        float B_col = get_linear_index(output_dims, k, j);
        subsum += A_row * B_col;
    
    }
    int idx_lin = get_linear_index(output_dims, i, j);
    output[idx_lin] = subsum;
}

std::vector<std::vector<float>> matmul_cuda(std::vector<std::vector<float>> A, std::vector<std::vector<float>> B){
   
    is_well_formed(A); is_well_formed(B); 
    check_compatibility(A, B);
    
    auto A_dims = get_dims(A);
    auto B_dims = get_dims(B);

    auto A_flattened = flatten_2D_matrix(A);
    auto B_flattened = flatten_2D_matrix(B);

    // Copy ctl structure and input matrices into linear buffer
    ctlInput control = {A_flattened.second, B_flattened.second, A_dims.second*B_dims.first};
    ctlInput *control_gpu;
    size_t mem_ctlIn = sizeof(ctlInput);
    cudaMalloc(&control_gpu, mem_ctlIn);
    cudaMemcpy(&control_gpu, &control, mem_ctlIn, cudaMemcpyHostToDevice);
    
    // Create linear buffer for matrix A
    float *A_arr;
    size_t mem_A = sizeof(float) * A_flattened.second;
    cudaMalloc(&A_arr, mem_A);
    cudaMemcpy(A_arr, A_flattened.first, mem_A, cudaMemcpyHostToDevice);

    // Create linear buffer for matrix B
    float *B_arr;
    size_t mem_B = sizeof(float) * B_flattened.second;
    cudaMalloc(&B_arr, mem_B);
    cudaMemcpy(B_arr, B_flattened.first, mem_B, cudaMemcpyHostToDevice);

    // Create output memory buffer on GPU
    float *output_arr;
    size_t mem_c = sizeof(float) * A_dims.second*B_dims.first;
    cudaMalloc(&output_arr, mem_c);

    int num_blocks = 1;
    dim3 threads_per_block(A_dims.second, B_dims.first);
    matmul_GPU<<<num_blocks, threads_per_block>>>(control_gpu, A_arr, B_arr, output_arr);

    // Get computed matrix off GPU
    auto C_arr = (float *)malloc(mem_c);
    cudaMemcpy(C_arr, &output_arr, mem_c, cudaMemcpyDeviceToHost);
    std::pair<float *, int> C_flattened = std::make_pair(C_arr, mem_c);
    auto C_dims = std::make_pair(A_dims.second, B_dims.first);

    std::vector<std::vector<float>> C = unflatten_1D_array(C_flattened, C_dims);

    return C;

}

