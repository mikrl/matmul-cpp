#include "extern/pybind11/include/pybind11/pybind11.h"
#include "matmul.h"

namespace py = pybind11;

PYBIND11_MODULE(matmul_simple, module){
    module.def("matmul_simple", &matmul_simple, "A function to multiply two matrices");

}

PYBIND11_MODULE(matmul_cuda, module){
    module.def("matmul_simple", &matmul_simple, "A function to multiply two matrices on the GPU");

}