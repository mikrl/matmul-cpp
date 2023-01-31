#include<iostream>
#include<vector>

#include "matmul.h"
#include "matrix_utils.h"

int main(){
    std::vector<std::vector<float>> A = {{1, 0, 0},{0, 1, 0},{0, 0, 1}};
    std::vector<std::vector<float>> B = {{2, 3, 4},{5, 4, 2},{1, 2, 3}};
    print_matrix(A);
    std::cout << '\n';
    print_matrix(B);
    std::cout << '\n';
    auto C = matmul_simple(A, B);
    print_matrix(C);
    return 0;
}

