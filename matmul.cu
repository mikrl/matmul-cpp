#define ARRAYSIZE(arr) (sizeof(arr) / sizeof(arr[0]))

#include<vector>
#include "matrix_utils.h"


struct ctlInput{
    int matrix_a_size;
    int matrix_b_size;
    int matrix_c_size;
};

__global__ void matmul_GPU(void *ctl_inputs, float *output){
    /* Simple matmul routine for a single GPU core.
       Assumes that input tensors have the correct dimensions for a matrix multiplication.
       Assumes output tensor has the correct dimension for a matrix multiplication. 
    */ 
    extern __shared__ float* memory[];
    int i = threadIdx.x;
    int j = threadIdx.y;

    // int B_rows = count_rows(B); 
    int B_rows = ARRAYSIZE(B);

    float subsum = 0;
    for(int k=0; k<B_rows; k++) subsum+=A[i][k]*B[k][j];
    C[i][j] = subsum;
}

std::vector<std::vector<float>> matmul_cuda(std::vector<std::vector<float>> A, std::vector<std::vector<float>> B){
   
    is_well_formed(A); is_well_formed(B); 
    check_compatibility(A, B);
    
    auto A_dims = get_dims(A);
    auto B_dims = get_dims(B);

    auto A_flattened = flatten_2D_matrix(A);
    auto B_flattened = flatten_2D_matrix(B);

    ctlInput mem = {A_flattened.second, B_flattened.second, A_dims.second*B_dims.first};

    size_t mem_ctlIn = sizeof(ctlInput);
    size_t mem_A = sizeof(float) * mem.matrix_a_size;
    size_t mem_B = sizeof(float) * mem.matrix_b_size;

    // Define linear buffer to move data to GPU
    size_t mem_cudabuf = mem_ctlIn + mem_A + mem_B;
    void *cuda_buffer = malloc(mem_cudabuf);

    // Copy ctl structure and input matrices into linear buffer
    memcpy(&cuda_buffer, &mem, mem_ctlIn);
    memcpy(&cuda_buffer+mem_ctlIn, A_flattened.first, mem_A);
    memcpy(&cuda_buffer+mem_ctlIn+mem_A, B_flattened.first, mem_B);

    // Copy memory buffer to GPU
    void *memGPU;
    cudaMalloc(&memGPU, mem_cudabuf);
    cudaMemcpy(memGPU, &cuda_buffer, mem_cudabuf, cudaMemcpyHostToDevice);

    // Create output memory buffer on GPU
    float *output_arr;
    size_t mem_c = sizeof(float) * A_dims.second*B_dims.first;
    cudaMalloc(&output_arr, mem_c);

    int num_blocks = 1;
    dim3 threads_per_block(A_dims.second, B_dims.first);
    matmul_GPU<<<num_blocks, threads_per_block>>>(memGPU, output_arr);

    // Get computed matrix off GPU
    auto C_arr = (float *)malloc(mem_c);
    cudaMemcpy(C_arr, &memGPU, mem_c, cudaMemcpyDeviceToHost);
    std::pair<float *, int> C_flattened = std::make_pair(C_arr, mem_c);
    auto C_dims = std::make_pair(A_dims.second, B_dims.first);

    std::vector<std::vector<float>> C = unflatten_1D_array(C_flattened, C_dims);

    return C;

}

