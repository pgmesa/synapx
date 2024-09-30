#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "tensor.hpp"

namespace py = pybind11;

// Helper function to convert Python object to DataType
DataType py_to_dtype(const py::object& dtype) {
    std::string dtype_str = py::str(dtype);
    if (dtype_str == "uint8") return DataType::UINT8;
    if (dtype_str == "int32") return DataType::INT32;
    if (dtype_str == "float32") return DataType::FLOAT32;

    throw std::invalid_argument("Unsupported data type: " + dtype_str);
}

// Templated function to bind the Tensor class
template<typename T>
void bind_tensor(py::module& m, const std::string& class_name) {
    py::class_<Tensor<T>>(m, class_name.c_str())
        .def(py::init<>())
        .def_readonly("numel", &Tensor<T>::numel)
        .def_readonly("shape", &Tensor<T>::shape)
        .def_readonly("ndim", &Tensor<T>::ndim)
        .def_readonly("dtype", &Tensor<T>::dtype)
        .def_readonly("strides", &Tensor<T>::strides)
        .def("__repr__", &Tensor<T>::to_string)
        //.def("__getitem__", &Tensor<T>::operator[])
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self * py::self)
        .def(py::self *= py::self)
        .def(py::self + double())
        .def(py::self += double())
        .def(py::self * double())
        .def(py::self *= double())
        .def("view", &Tensor<T>::view)
        .def("expand", &Tensor<T>::expand)
        .def("broadcast_to", &Tensor<T>::broadcast_to)
        .def("squeeze", &Tensor<T>::squeeze)
        .def("unsqueeze", &Tensor<T>::unsqueeze)
        // Use @ operator for matmul
        .def("__matmul__", static_cast<Tensor<T> (Tensor<T>::*)(const Tensor<T>&) const>(&Tensor<T>::matmul))
        .def_static("matmul", static_cast<Tensor<T> (*)(const Tensor<T>&, const Tensor<T>&)>(&Tensor<T>::matmul))
        .def_static("empty", [](const std::vector<int>& shape) {
            return Tensor<T>::empty(shape);
        })
        .def_static("full", [](const std::vector<int>& shape, double value) {
            return Tensor<T>::full(shape, value);
        })
        .def_static("ones", &Tensor<T>::ones)
        .def_static("zeros", &Tensor<T>::zeros);
}

PYBIND11_MODULE(synapx_c, m) {
    m.doc() = "pybind11 plugin for Tensor class";

    // Bind each instantiation of Tensor for the supported data types
    bind_tensor<uint8>(m, "TensorUInt8");
    bind_tensor<int32>(m, "TensorInt32");
    bind_tensor<float32>(m, "TensorFloat32");

    // Also bind the DataType enum
    py::enum_<DataType>(m, "DataType")
        .value("UINT8", DataType::UINT8)
        .value("INT32", DataType::INT32)
        .value("FLOAT32", DataType::FLOAT32);
}
