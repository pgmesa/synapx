#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "tensor.hpp"

namespace py = pybind11;

// Helper function to convert Python object to DataType
DataType py_to_dtype(const py::object& dtype) {
    // First check if dtype is already a DataType enum
    if (py::isinstance<DataType>(dtype)) {
        return dtype.cast<DataType>();
    }

    // If it's a string, convert it to a DataType
    std::string dtype_str = py::str(dtype);
    if (dtype_str == "uint8") return DataType::UINT8;
    if (dtype_str == "int32") return DataType::INT32;
    if (dtype_str == "float32") return DataType::FLOAT32;

    throw std::invalid_argument("Unsupported data type: " + dtype_str);
}


PYBIND11_MODULE(cpptensor, m) {
    m.doc() = "pybind11 example plugin for Tensor class";

    py::enum_<DataType>(m, "DataType")
        .value("UINT8", DataType::UINT8)
        .value("INT32", DataType::INT32)
        .value("FLOAT32", DataType::FLOAT32);

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def_readonly("numel", &Tensor::numel)
        .def_readonly("shape", &Tensor::shape)
        .def_readonly("ndim", &Tensor::ndim)
        .def_readonly("dtype", &Tensor::dtype)
        .def_readonly("strides", &Tensor::strides)
        .def("__repr__", &Tensor::to_string)
        .def("__getitem__", &Tensor::operator[])
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self * py::self)
        .def(py::self *= py::self)
        .def(py::self + double())
        .def(py::self += double())
        .def(py::self * double())
        .def(py::self *= double())
        // Use @ operator for matmul by binding to __matmul__
        .def("__matmul__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::matmul))
        // Static method version of matmul
        .def_static("matmul", static_cast<Tensor (*)(const Tensor&, const Tensor&)>(&Tensor::matmul))
        .def_static("empty", [](const std::vector<int>& shape, const py::object& dtype) {
            return Tensor::empty(shape, py_to_dtype(dtype));
        })
        .def_static("fill", [](const std::vector<int>& shape, const py::object& dtype, double value) {
            return Tensor::fill(shape, py_to_dtype(dtype), value);
        })
        .def_static("ones", [](const std::vector<int>& shape, const py::object& dtype) {
            return Tensor::ones(shape, py_to_dtype(dtype));
        })
        .def_static("zeros", [](const std::vector<int>& shape, const py::object& dtype) {
            return Tensor::zeros(shape, py_to_dtype(dtype));
        });
}
