A simple handrolled implementation of matrix multiplication in C++ and Python bindings

- C++ unit tests with gtest
- Python bindings with pybind11
- Python unit tests with unittest
# Build
`mkdir build`
`g++ -o ./build/main main.cpp matrix_utils.cpp matmul.cpp -lgtest -lpthread`
```g++ -o ./build/test matmul_test.cpp matrix_utils.cpp matmul.cpp -lgtest -lpthread```
## Python Bindings
`mkdir bindings`
```g++ -O3 -Wall -shared -std=c++17 -fPIC -I extern/pybind11/include/pybind11 matmul_binding.cpp matmul.cpp matrix_utils.cpp -o bindings/matmul_handrolled.so `python3-config --cflags --ldflags` ```

#Test
`./build/test`
