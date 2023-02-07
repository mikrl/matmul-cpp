A simple handrolled implementation of matrix multiplication in C++ and Python bindings

- C++ unit tests with gtest
- Python bindings with pybind11
- Python unit tests with unittest
# Build
`mkdir build`

`cd build`

`cmake ..`

`Make`

# Test C++ and CUDA routines
In the build dir:

`ctest`

If further granularity is required, the individual test executables can be called directly:

`matmul_test && matutils_test`

## Python Bindings

`mkdir bindings`

```g++ -O3 -Wall -shared -std=c++17 -fPIC -I extern/pybind11/include/pybind11 matmul_binding.cpp matmul.cpp matrix_utils.cpp -o bindings/matmul_handrolled.so `python3-config --cflags --ldflags` ```

# Python Tests
In virtualenv with numpy, and after building bindings

`python3 -m unittest discover -s tests`
