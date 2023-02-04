#ifndef MAT_UTILS
#define MAT_UTILS

#include<vector>

void print_matrix(std::vector<std::vector<float>> matrix);

int count_rows(std::vector<std::vector<float>> A);

int count_cols(std::vector<std::vector<float>> A);

std::pair<int, int> get_dims(std::vector<std::vector<float>> matrix);

std::vector<std::vector<float>> allocate_empty_matrix(int m_rows, int n_cols);

void is_well_formed(std::vector<std::vector<float>> matrix);

void check_compatibility(std::vector<std::vector<float>> A, std::vector<std::vector<float>> B);

std::pair<float*, int> flatten_2D_matrix(std::vector<std::vector<float>> matrix);

std::vector<std::vector<float>> unflatten_1D_array(std::pair<float*, int> flat_array, std::pair<int, int> dims);

#endif