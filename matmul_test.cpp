#include <gtest/gtest.h>
#include "matmul.h"

TEST(MatmulTest, TestCase1) {
    std::vector<std::vector<float>> matA = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::vector<std::vector<float>> matB = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    std::vector<std::vector<float>> expected = {{30, 24, 18}, {84, 69, 54}, {138, 114, 90}};
    std::vector<std::vector<float>> result = matmul(matA, matB);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_EQ(expected[i][j], result[i][j]);
        }
    }
}

TEST(MatmulTest, TestCase2) {
    std::vector<std::vector<float>> matA = {{1, 2}, {3, 4}};
    std::vector<std::vector<float>> matB = {{5, 6}, {7, 8}};
    std::vector<std::vector<float>> expected = {{19, 22}, {43, 50}};
    std::vector<std::vector<float>> result = matmul(matA, matB);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            EXPECT_EQ(expected[i][j], result[i][j]);
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}