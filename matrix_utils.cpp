#define ARRAYSIZE(arr) (sizeof(arr) / sizeof(arr[0]))
#include<iostream>
#include<vector>

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

int count_rows(std::vector<std::vector<float>> A){
    return A.size();
}

int count_cols(std::vector<std::vector<float>> A){
    return A[0].size();
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

void check_compatibility(std::vector<std::vector<float>> A, std::vector<std::vector<float>> B){
    int A_cols = count_cols(A);
    
    int B_rows = count_rows(B);

    if (A_cols != B_rows) error("Matrix A columns should match Matrix B rows.");

}

float** matrix_to_arr(std::vector<std::vector<float>> A) {
    int m = A.size();
    int n = A[0].size();
    float** arr = new float*[m];
    int i, j;
    for (i = 0; i < m; i++) {
        arr[i] = new float[n];
        for (j = 0; j < n; j++) {
            arr[i][j] = A[i][j];
        }
    }

    return arr;
}

std::vector<std::vector<float>> arr_to_matrix(float **A, int rows, int cols) {

     std::vector<std::vector<float>> matrix(rows);
     for (int i = 0; i < rows; i++) {
         matrix[i].resize(cols);
     for (int j = 0; j < cols; j++) {
       matrix[i][j] = A[i][j];
     }
   }
  return matrix;
}