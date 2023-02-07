cmake_minimum_required(VERSION 3.13)

# set CUDA version
set(CUDA_VERSION 11.0)

# set cuda architectures to build for
set(CUDA_ARCH "-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75")

# set project name
project(MatMul CXX CUDA)

# specify compiler flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# include cuda libraries
find_package(CUDA ${CUDA_VERSION} REQUIRED)

# specify include directories
include_directories(include)
include_directories(/usr/local/cuda/include)

# add CUDA library
cuda_add_library(matmul_cuda
    src/matmul.cu
)

# add C++ libraries
add_library(matrix_utils src/matrix_utils.cpp)
add_library(matmul src/matmul.cpp)

# specify libraries to link
target_link_libraries(matrix_utils PRIVATE matmul_cuda)
target_link_libraries(matmul PRIVATE matrix_utils)

# specify executable
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE matmul)

# set build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# specify cuda flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo ${CUDA_ARCH}")

# enable tests
enable_testing()

# add C++ test executable for matmul tests
add_executable(matmul_test test/matmul_test.cpp)
target_link_libraries(matmul_test PRIVATE matmul)
add_test(NAME matmul_test COMMAND matmul_test)

# add C++ test executable for matutil tests
add_executable(matutils_test test/matutils_test.cpp)
target_link_libraries(matutils_test PRIVATE matmul)
add_test(NAME matutils_test COMMAND matutils_test)

# add gtest library
find_package(GTest REQUIRED)
target_link_libraries(matmul_test PRIVATE GTest::GTest GTest::Main)
target_link_libraries(matutils_test PRIVATE GTest::GTest GTest::Main)

