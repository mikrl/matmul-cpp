cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA
find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)

# GTest
find_package(GTest REQUIRED)
enable_testing()

# Find PyBind11
set(PYBIND11_CPP_STANDARD -std=c++11)
add_subdirectory(extern/pybind11)

# Include Directories
include_directories(
  include
  extern/pybind11/include
  ${GTEST_INCLUDE_DIRS}
)

# Library Sources
set(CUDA_SOURCES
  src/matrix_utils.cu
  src/matmul.cu
)

set(CPP_SOURCES
  src/matrix_utils.cpp
  src/matmul.cpp
  src/matmul_binding.cpp
)

# Build CUDA Libraries
cuda_add_library(cuda_lib ${CUDA_SOURCES})

# Build CPP Libraries
add_library(cpp_lib ${CPP_SOURCES})
target_link_libraries(cpp_lib
  cuda_lib
  pybind11::pybind11
)

# Main Executable
cuda_add_executable(main
  src/main.cpp
)

target_link_libraries(main
  cpp_lib
)

# Test Executable
add_executable(tests
  test/matmul_test.cpp
  test/matutils_test.cpp
)

target_link_libraries(tests
  cpp_lib
  ${GTEST_BOTH_LIBRARIES}
)

# Add Tests
add_test(Test tests)
