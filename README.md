A simple handrolled implementation of matrix multiplication in C++ and Python bindings

- C++ unit tests with gtest
- Python bindings with pybind11
- Python unit tests with unittest
# Build
`g++ -o ./build/main main.cpp matrix_utils.cpp matmul.cpp -lgtest -lpthread`

## Python Bindings
```g++ -O3 -Wall -shared -std=c++17 -fPIC -I extern/pybind11/include/pybind11 matmul_binding.cpp matmul.cpp matrix_utils.cpp -o bindings/matmul_handrolled.so `python3-config --cflags --ldflags` ```

# Tests
To run C++ tests

```g++ -o ./build/test matmul_test.cpp matrix_utils.cpp matmul.cpp -lgtest -lpthread && ./build/test```

In virtualenv with numpy, and after building bindings

`python3 -m unittest discover -s tests`
