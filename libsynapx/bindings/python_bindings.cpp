

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/torch.h>
#include <synapx/tensor.hpp>


namespace py = pybind11;


#if defined(_MSC_VER) && !defined(ssize_t)
typedef std::ptrdiff_t ssize_t;
#endif


py::array tensor_to_numpy(const synapx::Tensor& tensor) {
    // Get the underlying torch::Tensor
    const auto& torch_tensor = tensor.data;

    // Ensure the tensor is contiguous
    auto contiguous_tensor = torch_tensor.contiguous();

    // Get the tensor's data pointer, shape, and stride
    void* data_ptr = contiguous_tensor.data_ptr();
    std::vector<ssize_t> shape(contiguous_tensor.sizes().begin(), contiguous_tensor.sizes().end());
    std::vector<ssize_t> strides(contiguous_tensor.strides().begin(), contiguous_tensor.strides().end());

    // Convert strides from element-based to byte-based
    for (auto& stride : strides) {
        stride *= contiguous_tensor.element_size();
    }

    // Determine the NumPy data type
    py::dtype dtype;
    switch (contiguous_tensor.scalar_type()) {
        case torch::kFloat32: dtype = py::dtype::of<float>(); break;
        case torch::kFloat64: dtype = py::dtype::of<double>(); break;
        case torch::kInt32: dtype = py::dtype::of<int32_t>(); break;
        case torch::kInt64: dtype = py::dtype::of<int64_t>(); break;
        default:
            throw std::runtime_error("Unsupported tensor data type for conversion to NumPy");
    }

    // Create and return the NumPy array (no memory copy)
    return py::array(dtype, shape, strides, data_ptr, py::capsule(data_ptr, [](void* ptr) {
        // Custom deleter for the memory, if needed
        // Note: PyTorch handles memory deallocation, so no action needed here
    }));
}


PYBIND11_MODULE(_C, m) {
    m.doc() = "Synapx tensor operations";

    py::class_<synapx::Tensor>(m, "Tensor")
        .def(py::init<const torch::Tensor&>())
        .def("numel", &synapx::Tensor::numel)
        .def("ndim", &synapx::Tensor::ndim)
        .def("matmul", &synapx::Tensor::matmul)
        .def("numpy", [](const synapx::Tensor& tensor) {
            return tensor_to_numpy(tensor);
        }, "Convert Tensor to NumPy array");

    m.def("ones", [](py::list shape) {
        std::vector<int64_t> dims;
        for (auto item : shape) {
            dims.push_back(py::cast<int64_t>(item));
        }
        return synapx::Tensor(torch::ones(dims));
    }, "Create a tensor filled with ones");

    m.def("zeros", [](py::list shape) {
        std::vector<int64_t> dims;
        for (auto item : shape) {
            dims.push_back(py::cast<int64_t>(item));
        }
        return synapx::Tensor(torch::zeros(dims));
    }, "Create a tensor filled with zeros");

    m.def("matmul", [](synapx::Tensor t1, synapx::Tensor t2) {
        return t1.matmul(t2);
    }, "Matmul between two tensors");

    m.def("from_numpy", [](py::array array) {
        // Ensure the input is a contiguous array
        if (!array.flags() & py::array::c_style) {
            throw std::runtime_error("Input array must be contiguous");
        }

        // Get the pointer to the data and its shape information
        void* data_ptr = array.mutable_data(); // mutable_data() gives writeable pointer
        std::vector<int64_t> shape(array.shape(), array.shape() + array.ndim());

        // Infer the data type
        torch::Dtype dtype;
        if (array.dtype().is(py::dtype::of<float>())) {
            dtype = torch::kFloat32;
        } else if (array.dtype().is(py::dtype::of<double>())) {
            dtype = torch::kFloat64;
        } else if (array.dtype().is(py::dtype::of<int32_t>())) {
            dtype = torch::kInt32;
        } else if (array.dtype().is(py::dtype::of<int64_t>())) {
            dtype = torch::kInt64;
        } else {
            throw std::runtime_error("Unsupported NumPy data type");
        }

        // Create the PyTorch tensor from the NumPy array buffer
        auto tensor = torch::from_blob(data_ptr, shape, dtype);

        // Set the tensor to share memory with the NumPy array
        tensor = tensor.clone(); // Optional: Clone to decouple memory if needed

        return synapx::Tensor(tensor);
    }, "Create a tensor from numpy array");
}
