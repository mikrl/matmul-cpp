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

void check_matmul_compatibility(int A_cols, int B_rows){
        if (A_cols != B_rows) throw;
}