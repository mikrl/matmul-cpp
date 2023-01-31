#ifndef MATMUL
#define MATMUL

#include<vector>

std::vector<std::vector<float>> matmul_simple(std::vector<std::vector<float>> A, std::vector<std::vector<float>> B);

std::vector<std::vector<float>> matmul_cuda(std::vector<std::vector<float>> A, std::vector<std::vector<float>> B);

#endif