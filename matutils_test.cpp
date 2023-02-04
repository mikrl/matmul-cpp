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

// Test matrix_to_arr and arr_to_matrix functions
TEST(MatrixUtilsTest, MatrixArrConversion) {
  std::vector<std::vector<float>> A = {{1, 2, 3}, {4, 5, 6}};
  float** arr = matrix_to_arr(A);

  // Check conversion from matrix to array
  EXPECT_EQ(arr[0][0], 1);
  EXPECT_EQ(arr[0][1], 2);
  EXPECT_EQ(arr[0][2], 3);
  EXPECT_EQ(arr[1][0], 4);
  EXPECT_EQ(arr[1][1], 5);
  EXPECT_EQ(arr[1][2], 6);

  // Check conversion from array to matrix
  std::vector<std::vector<float>> B = arr_to_matrix(arr, A.size(), A[0].size());
  EXPECT_EQ(B[0][0], 1);
  EXPECT_EQ(B[0][1], 2);
  EXPECT_EQ(B[0][2], 3);
  EXPECT_EQ(B[1][0], 4);
  EXPECT_EQ(B[1][1], 5);
  EXPECT_EQ(B[1][2], 6);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}