#define ARRAYSIZE(arr) (sizeof(arr) / sizeof(arr[0]))
#include<iostream>
#include<vector>
#include <stdlib.h>

void print_matrix(std::vector<std::vector<float>> matrix){
    for (auto& row : matrix){
        std::cout<<'|';
        for (auto& col : row){
            std::cout << col << '\t';
        }
        std::cout<<'|';
        std::cout << '\n';
    } 

}

int count_rows(std::vector<std::vector<float>> matrix){
    return matrix.size();
}

int count_cols(std::vector<std::vector<float>> matrix){
    return matrix[0].size();
}

std::pair<int, int> get_dims(std::vector<std::vector<float>> matrix){
    std::pair dims = {count_rows(matrix), count_cols(matrix)};
    return dims;

}

std::vector<std::vector<float>> allocate_empty_matrix(int m_rows, int n_cols){
    std::vector<std::vector<float>> matrix;

    for (int col_idx = 0; col_idx < n_cols; col_idx++){
        std::vector<float> row(m_rows);          
        matrix.push_back(row);
    }
    return matrix;
}



inline void error(const std::string& s){
    throw std::runtime_error(s);
}


void is_well_formed(std::vector<std::vector<float>> matrix){
    int cols = matrix[0].size();
    for (auto & row : matrix){
        if(row.size() != cols){
            error("Matrix has rows of different widths");
        }
    }
}

void check_compatibility(std::vector<std::vector<float>> A, std::vector<std::vector<float>> B){
    int A_cols = count_cols(A);
    int B_rows = count_rows(B);

    if (A_cols != B_rows) error("Matrix A columns should match Matrix B rows.");

}

std::pair<float*, int> flatten_2D_matrix(std::vector<std::vector<float>> matrix){

    auto dims = get_dims(matrix);
    size_t flat_size = dims.first * dims.second;

    float *flat_arr = (float *) malloc(flat_size*sizeof(float));
 
    for (int i =0, pos=0; i<dims.first; i++){
        for (int j =0; j<dims.second; j++){
            flat_arr[pos++] = matrix[i][j];
        }
    }
 
    std::pair<float*, int> array_1D = {flat_arr, flat_size};
    return array_1D;

}

std::vector<std::vector<float>> unflatten_1D_array(std::pair<float*, int> flat_array, std::pair<int, int> dims){
    auto matrix = allocate_empty_matrix(dims.first, dims.second);
    
    auto flat_arr = flat_array.first;

    auto row_stride = dims.second;

    for (int pos=0, row_idx=0; pos<flat_array.second; pos+=row_stride, row_idx++){
        for(int col_idx=0; col_idx<row_stride; col_idx++)
            matrix[row_idx][col_idx] = flat_arr[pos+col_idx];
    }

    return matrix;
}

int get_linear_index(int matrix_dims[2], int row_idx, int col_idx){
    auto n_cols = matrix_dims[1];

    return row_idx*n_cols + col_idx;
}


int get_row_idx(int idx_linear, int n_cols){
    return idx_linear / n_cols;
}

int get_col_idx(int idx_linear, int n_cols){
    return idx_linear % n_cols;
}
