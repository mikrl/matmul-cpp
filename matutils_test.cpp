#include <gtest/gtest.h>

#include "matrix_utils.h"

TEST(MatrixUtilsTest, CountRows) {
    std::vector<std::vector<float>> matrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int expected_rows = 3;
    int actual_rows = count_rows(matrix);
    EXPECT_EQ(expected_rows, actual_rows);
}

TEST(MatrixUtilsTest, CountCols) {
    std::vector<std::vector<float>> matrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int expected_cols = 3;
    int actual_cols = count_cols(matrix);
    EXPECT_EQ(expected_cols, actual_cols);
}



TEST(MatrixUtilsTest, CheckCompatibility) {
    std::vector<std::vector<float>> A = {{1, 2, 3}, {4, 5, 6}};
    std::vector<std::vector<float>> B = {{1, 2, 3, 4}, {5, 6, 7, 8}};
    EXPECT_THROW(check_compatibility(A, B), std::runtime_error);
}

TEST(MatrixFunctions, allocateUninitializedMatrix) {
  std::vector<std::vector<float>> A = allocate_empty_matrix(3, 3);
  EXPECT_EQ(count_rows(A), 3);
  EXPECT_EQ(count_cols(A), 3);
}

// Test check_matmul_compatibility function
TEST(MatrixUtilsTest, CheckMatmulCompatibility) {
  std::vector<std::vector<float>> A = {{1, 2, 3}, {4, 5, 6}};
  std::vector<std::vector<float>> B = {{1, 2}, {3, 4}, {5, 6}};

  // Test compatibility
  EXPECT_NO_THROW(check_compatibility(A, B));

  // Test incompatibility
  B.clear();
  B = {{1, 2, 3}};
  EXPECT_THROW(check_compatibility(A, B), std::runtime_error);
}

// Test is_well_formed function
TEST(MatrixUtilsTest, IsWellFormed) {
  std::vector<std::vector<float>> matrix_wf = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  std::vector<std::vector<float>> matrix_nwf = {{1, 2, 3}, {4, 5}, {9}};

  // Test well formed matrix
  EXPECT_NO_THROW(is_well_formed(matrix_wf));

  // Test ill formed matrix
  EXPECT_THROW(is_well_formed(matrix_nwf), std::runtime_error);
}

// Test flatten_2D_matrix function
TEST(MatrixUtilsTest, CheckFlatten) {
  std::vector<std::vector<float>> matrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  float arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  auto flattened = flatten_2D_matrix(matrix);
  float *new_arr = flattened.first;
  
  EXPECT_EQ(new_arr[0], 1);
  EXPECT_EQ(new_arr[1], 2);
  EXPECT_EQ(new_arr[2], 3);
  EXPECT_EQ(new_arr[3], 4);
  EXPECT_EQ(new_arr[4], 5);
  EXPECT_EQ(new_arr[5], 6);
  EXPECT_EQ(new_arr[6], 7);
  EXPECT_EQ(new_arr[7], 8);
  EXPECT_EQ(new_arr[8], 9);

  free((void*) new_arr);

  int new_arr_size = flattened.second;
  EXPECT_EQ(new_arr_size, 9);
}

// Test unflatten_1D_array function
TEST(MatrixUtilsTest, CheckUnflatten) {
  float arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<std::vector<float>> matrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

  std::pair<float *, int> flat_array = {arr, 9};
  std::pair<int, int> dims = get_dims(matrix);

  auto unflattened_matrix = unflatten_1D_array(flat_array, dims);

  EXPECT_EQ(unflattened_matrix[0][0], 1);
  EXPECT_EQ(unflattened_matrix[0][1], 2);
  EXPECT_EQ(unflattened_matrix[0][2], 3);
  EXPECT_EQ(unflattened_matrix[1][0], 4);
  EXPECT_EQ(unflattened_matrix[1][1], 5);
  EXPECT_EQ(unflattened_matrix[1][2], 6);
  EXPECT_EQ(unflattened_matrix[2][0], 7);
  EXPECT_EQ(unflattened_matrix[2][1], 8);
  EXPECT_EQ(unflattened_matrix[2][2], 9);
}

// // Test matrix_to_arr and arr_to_matrix functions
// TEST(MatrixUtilsTest, MatrixArrConversion) {
//   std::vector<std::vector<float>> A = {{1, 2, 3}, {4, 5, 6}};
//   float** arr = matrix_to_arr(A);

//   // Check conversion from matrix to array
//   EXPECT_EQ(arr[0][0], 1);
//   EXPECT_EQ(arr[0][1], 2);
//   EXPECT_EQ(arr[0][2], 3);
//   EXPECT_EQ(arr[1][0], 4);
//   EXPECT_EQ(arr[1][1], 5);
//   EXPECT_EQ(arr[1][2], 6);

//   // Check conversion from array to matrix
//   std::vector<std::vector<float>> B = arr_to_matrix(arr, A.size(), A[0].size());
//   EXPECT_EQ(B[0][0], 1);
//   EXPECT_EQ(B[0][1], 2);
//   EXPECT_EQ(B[0][2], 3);
//   EXPECT_EQ(B[1][0], 4);
//   EXPECT_EQ(B[1][1], 5);
//   EXPECT_EQ(B[1][2], 6);
// }

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}