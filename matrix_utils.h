#include<vector>

void print_matrix(std::vector<std::vector<float>> matrix);

int count_rows(std::vector<std::vector<float>> A);

int count_cols(std::vector<std::vector<float>> A);

std::vector<std::vector<float>> allocate_empty_matrix(int m_rows, int n_cols);

void check_matmul_compatibility(int A_cols, int B_rows);