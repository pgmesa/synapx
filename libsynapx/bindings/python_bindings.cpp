
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <torch/torch.h>
#include <synapx/tensor.hpp>


namespace py = pybind11;


#if defined(_MSC_VER) && !defined(ssize_t)
typedef std::ptrdiff_t ssize_t;
#endif

torch::Tensor numpy_to_torch(py::array array) {
    // Ensure the input is a contiguous array
    if (!(array.flags() & py::array::c_style)) {
        // Make it contiguous if it's not
        array = py::array::ensure(array, py::array::c_style);
    }

    // Get the pointer to the data and its shape information
    void* data_ptr = array.mutable_data();
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
    } else if (array.dtype().is(py::dtype::of<uint8_t>())) {
        dtype = torch::kUInt8;
    } else if (array.dtype().is(py::dtype::of<int8_t>())) {
        dtype = torch::kInt8;
    } else if (array.dtype().is(py::dtype::of<int16_t>())) {
        dtype = torch::kInt16;
    } else {
        throw std::runtime_error("Unsupported NumPy data type for tensor conversion");
    }

    // Create the PyTorch tensor from the NumPy array buffer
    auto tensor = torch::from_blob(data_ptr, shape, dtype);
    return tensor.clone(); // Clone to decouple memory
}


py::array synapx_to_numpy(const synapx::Tensor& tensor) {
    // Get the underlying torch::Tensor
    const auto& torch_tensor = tensor.data();

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
        case torch::kUInt8: dtype = py::dtype::of<uint8_t>(); break;
        case torch::kInt8: dtype = py::dtype::of<int8_t>(); break;
        case torch::kInt16: dtype = py::dtype::of<int16_t>(); break;
        default:
            throw std::runtime_error("Unsupported tensor data type for conversion to NumPy");
    }

    // Create and return the NumPy array (no memory copy)
    return py::array(dtype, shape, strides, data_ptr, py::capsule(data_ptr, [](void* ptr) {
        // Custom deleter for the memory, if needed
        // Note: PyTorch handles memory deallocation, so no action needed here
    }));
}

// Helper function to convert string to Device
synapx::Device string_to_device(const std::string& device_str) {
    if (device_str == "cpu" || device_str == "CPU") {
        return synapx::Device::CPU();
    } else {
        throw std::runtime_error("Unsupported device: " + device_str + ". Use 'cpu'");
    }
}

// Helper function to convert data to torch::Tensor
torch::Tensor pydata_to_torch(py::object data) {
    torch::Tensor tensor;
    
    if (py::isinstance<py::list>(data)) {
        // Convert Python list to tensor
        py::list py_list = py::cast<py::list>(data);
        
        // Convert to NumPy first for easier handling
        py::array numpy_array = py::array(py_list);
        tensor = numpy_to_torch(numpy_array);
        
    } else if (py::isinstance<py::array>(data)) {
        // Convert NumPy array to tensor
        py::array numpy_array = py::cast<py::array>(data);
        tensor = numpy_to_torch(numpy_array);
        
    } else {
        throw std::runtime_error("data must be a Python list or NumPy array");
    }
    
    return tensor;
}


PYBIND11_MODULE(_C, m) {
    m.doc() = "Synapx core C++ bindings";
    py::class_<synapx::Tensor>(m, "Tensor")
        .def("numel", &synapx::Tensor::numel)
        .def("dim", &synapx::Tensor::dim)
        .def_property_readonly("shape", &synapx::Tensor::shape)
        .def("add", &synapx::Tensor::add, py::arg("other"))
        .def("mul", &synapx::Tensor::mul, py::arg("other"))
        .def("matmul", &synapx::Tensor::matmul, py::arg("other"))
        .def("__add__", &synapx::Tensor::add, py::arg("other"))
        .def("__mul__", &synapx::Tensor::mul, py::arg("other"))
        .def("__matmul__", &synapx::Tensor::matmul, py::arg("other"))
        .def("requires_grad", &synapx::Tensor::requires_grad)
        .def("is_leaf", &synapx::Tensor::is_leaf)
        .def("retain_grad", &synapx::Tensor::retain_grad)
        .def("retains_grad", &synapx::Tensor::retains_grad)
        .def_property_readonly("grad", [](const synapx::Tensor& self) -> py::object {
            torch::Tensor grad_tensor = self.grad();
            if (grad_tensor.defined()) {
                return py::cast(synapx::Tensor(grad_tensor));
            } else {
                return py::none();
            }
        }, "Union[None, synapx.Tensor]: Gradient tensor or None")
        .def("backward", [](synapx::Tensor& self, py::object grad) {
            if (grad.is_none()) {
                self.backward();
            } else if (py::isinstance<synapx::Tensor>(grad)) {
                synapx::Tensor grad_tensor = py::cast<synapx::Tensor>(grad);
                self.backward(grad_tensor.data());
            }
        }, py::arg("grad") = py::none(), 
           "Union[None, synapx.Tensor]: Computes the gradient of current tensor w.r.t. graph leaves.")
        .def("numpy", [](const synapx::Tensor& tensor) {
            return synapx_to_numpy(tensor);
        });
    
    m.def("tensor", [](py::object data, bool requires_grad, std::string device) {
        synapx::Device dev = string_to_device(device);
        torch::Tensor torch_tensor = pydata_to_torch(data);
        return synapx::Tensor(torch_tensor, requires_grad, dev);
    }, py::arg("data"), py::arg("requires_grad") = false, py::arg("device") = "cpu");
    
    m.def("ones", [](py::list shape) {
        std::vector<int64_t> dims;
        for (auto item : shape) {
            dims.push_back(py::cast<int64_t>(item));
        }
        return synapx::Tensor(torch::ones(dims));
    }, py::arg("shape"));
    
    m.def("zeros", [](py::list shape) {
        std::vector<int64_t> dims;
        for (auto item : shape) {
            dims.push_back(py::cast<int64_t>(item));
        }
        return synapx::Tensor(torch::zeros(dims));
    }, py::arg("shape"));
    
    m.def("add", [](synapx::Tensor t1, synapx::Tensor t2) {
        return t1.add(t2);
    }, py::arg("t1"), py::arg("t2"));
    
    m.def("mul", [](synapx::Tensor t1, synapx::Tensor t2) {
        return t1.mul(t2);
    }, py::arg("t1"), py::arg("t2"));
    
    m.def("matmul", [](synapx::Tensor t1, synapx::Tensor t2) {
        return t1.matmul(t2);
    }, py::arg("t1"), py::arg("t2"));
}
