
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtensor/xarray.hpp>
#include "_C/tensor.cpp"  // Include your Tensor implementation

namespace py = pybind11;

PYBIND11_MODULE(synapx_c, m) {
    py::class_<Tensor<float>>(m, "Tensor")
        .def(py::init<const std::vector<size_t>&>())  // Constructor with shape
        //.def("set_value", &Tensor<float>::setValue)  // Set value in tensor
        //.def("get_value", &Tensor<float>::getValue)  // Get value from tensor
        .def_static("zeros", &Tensor<float>::zeros)  // Create tensor of zeros
        .def_static("ones", &Tensor<float>::ones)    // Create tensor of ones
        //.def_static("full", &Tensor<float>::full)    // Create tensor of specific value
        .def("add", &Tensor<float>::add)             // Add two tensors
        .def("matmul", &Tensor<float>::matmul)       // Matrix multiplication
        .def("print", &Tensor<float>::print);        // Print tensor content
}
