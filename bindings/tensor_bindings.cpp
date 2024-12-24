
#include "../src/tensor.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <xtensor/xarray.hpp>

namespace py = pybind11;

PYBIND11_MODULE(synapx_c, m) {

    // Add module-level functions
    m.def("ones", [](const std::vector<size_t>& shape) {
        return Tensor<float>::ones(shape);
    }, "Create a tensor filled with ones");

    m.def("zeros", [](const std::vector<size_t>& shape) {
        return Tensor<float>::zeros(shape);
    }, "Create a tensor filled with zeros");

    m.def("from_numpy", [](py::array_t<float> numpy_array) {
        py::buffer_info info = numpy_array.request();
        if (info.ndim <= 0) {
            throw std::runtime_error("NumPy array must have at least one dimension");
        }
        std::vector<size_t> shape(info.shape.begin(), info.shape.end());
        Tensor<float> tensor(shape);
        std::memcpy(tensor.array.data(), info.ptr, info.size * sizeof(float));
        return tensor;
    }, "Create a tensor from a numpy array");

    py::class_<Tensor<float>>(m, "Tensor")
        .def(py::init<const std::vector<size_t>&>())  // Constructor with shape
        .def("add", &Tensor<float>::add)             // Add two tensors
        .def("mul", &Tensor<float>::mul)
        .def("matmul", &Tensor<float>::matmul)       // Matrix multiplication
        .def("numpy", [](Tensor<float>& self) {
            auto& xtensor_array = self.array;

            // Convert xtensor strides (elements) to NumPy strides (bytes)
            std::vector<py::ssize_t> strides_in_bytes(xtensor_array.strides().size());
            for (size_t i = 0; i < xtensor_array.strides().size(); ++i) {
                strides_in_bytes[i] = static_cast<py::ssize_t>(xtensor_array.strides()[i] * sizeof(float));
            } 

            return py::array_t<float>(
                xtensor_array.shape(),  // Shape of the array
                strides_in_bytes,  // Strides (in bytes) of the array
                xtensor_array.data()  // Pointer to the data
            );
        })
        ;
}
