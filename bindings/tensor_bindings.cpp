
#include "../src/tensor.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <xtensor/xarray.hpp>

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
        .def("to_numpy", [](Tensor<float>& self) {
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
        })  // Convert to NumPy array
        .def_static("from_numpy", [](py::array_t<float> numpy_array) {
            // Request a buffer from the NumPy array
            py::buffer_info info = numpy_array.request();

            if (info.ndim <= 0) {
                throw std::runtime_error("NumPy array must have at least one dimension");
            }

            // Convert shape to a vector<size_t>
            std::vector<size_t> shape(info.shape.begin(), info.shape.end());

            // Create a Tensor object
            Tensor<float> tensor(shape);

            // Copy data from NumPy array to the xtensor
            std::memcpy(tensor.array.data(), info.ptr, info.size * sizeof(float));

            return tensor;
        })  // Create Tensor from NumPy array
        .def("print", &Tensor<float>::print);        // Print tensor content
}
