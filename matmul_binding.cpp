#include "extern/pybind11/include/pybind11/pybind11.h"
#include "matmul.h"

namespace py = pybind11;

PYBIND11_MODULE(matmul_handrolled, module){
    module.def("matmul", &matmul, "A function to multiply two matrices");

}