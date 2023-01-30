import unittest

import numpy as np

import bindings.matmul_handrolled as matmul_handrolled

class TestMatMulMethods(unittest.TestCase):

    def test_matrix_mult(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        expected_result = np.array([[19, 22], [43, 50]])
        result = matmul_handrolled.matmul(A, B)
        np.testing.assert_array_almost_equal(result, expected_result)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()