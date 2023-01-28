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