#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <synapx/tensor.hpp>


namespace py = pybind11;

#if defined(_MSC_VER) && !defined(ssize_t)
typedef std::ptrdiff_t ssize_t;
#endif


// Converts a numpy.ndarray (py::array) → torch::Tensor
inline torch::Tensor numpy_to_torch(py::array array) {
    // Make sure it’s contiguous:
    if (!(array.flags() & py::array::c_style)) {
        array = py::array::ensure(array, py::array::c_style);
    }

    // Raw data pointer and shape:
    void* data_ptr = array.mutable_data();
    std::vector<int64_t> shape(array.shape(), array.shape() + array.ndim());

    // Figure out dtype:
    torch::Dtype dtype;
    if (array.dtype().is(py::dtype::of<float>()))            { dtype = torch::kFloat32; }
    else if (array.dtype().is(py::dtype::of<double>()))      { dtype = torch::kFloat64; }
    else if (array.dtype().is(py::dtype::of<int32_t>()))     { dtype = torch::kInt32; }
    else if (array.dtype().is(py::dtype::of<int64_t>()))     { dtype = torch::kInt64; }
    else if (array.dtype().is(py::dtype::of<uint8_t>()))     { dtype = torch::kUInt8; }
    else if (array.dtype().is(py::dtype::of<int8_t>()))      { dtype = torch::kInt8; }
    else if (array.dtype().is(py::dtype::of<int16_t>()))     { dtype = torch::kInt16; }
    else {
        throw std::runtime_error("Unsupported NumPy data type for tensor conversion");
    }

    // Create a tensor from the raw buffer, then clone so PyTorch owns its own memory:
    auto tens = torch::from_blob(data_ptr, shape, dtype);
    return tens.clone();
}

// Converts a torch::Tensor → numpy.ndarray
inline py::array torch_to_numpy(const torch::Tensor& torch_tensor) {
    auto contig = torch_tensor.contiguous();
    void* data_ptr = contig.data_ptr();
    std::vector<ssize_t> shape(contig.sizes().begin(), contig.sizes().end());
    std::vector<ssize_t> strides(contig.strides().begin(), contig.strides().end());

    // Convert strides (in elements) → strides in bytes:
    for (auto& st : strides) {
        st *= contig.element_size();
    }

    // Figure out numpy dtype:
    py::dtype dtype;
    switch (contig.scalar_type()) {
        case torch::kFloat32: dtype = py::dtype::of<float>();   break;
        case torch::kFloat64: dtype = py::dtype::of<double>();  break;
        case torch::kInt32:   dtype = py::dtype::of<int32_t>(); break;
        case torch::kInt64:   dtype = py::dtype::of<int64_t>(); break;
        case torch::kUInt8:   dtype = py::dtype::of<uint8_t>(); break;
        case torch::kInt8:    dtype = py::dtype::of<int8_t>();  break;
        case torch::kInt16:   dtype = py::dtype::of<int16_t>(); break;
        default:
            throw std::runtime_error("Unsupported tensor data type for conversion to NumPy");
    }

    // Return an array that (non‐copying) wraps the PyTorch buffer:
    return py::array(
        dtype,
        shape,
        strides,
        data_ptr,
        // Attach a capsule so PyTorch’s memory isn’t freed prematurely:
        py::capsule(data_ptr, [](void* p){ /* no-op; PyTorch owns it */ })
    );
}

// Convert a std::string (e.g. "cpu") → synapx::Device
inline synapx::Device string_to_device(const std::string& device_str) {
    if (device_str == "cpu" || device_str == "CPU") {
        return synapx::Device::CPU();
    } else {
        throw std::runtime_error("Unsupported device: " + device_str + ". Use 'cpu'");
    }
}

// Convert a PyObject (scalar, list, numpy.ndarray, torch.Tensor, or synapx.Tensor) → torch::Tensor
inline torch::Tensor pyobject_to_torch(py::object data) {
    torch::Tensor tensor;

    // Handle Python scalars (int, float, bool)
    if (py::isinstance<py::int_>(data)) {
        int64_t value = py::cast<int64_t>(data);
        tensor = torch::tensor(value);
    }
    else if (py::isinstance<py::float_>(data)) {
        double value = py::cast<double>(data);
        tensor = torch::tensor(value);
    }
    else if (py::isinstance<py::bool_>(data)) {
        bool value = py::cast<bool>(data);
        tensor = torch::tensor(value);
    }
    // If it's a Python list, first convert to NumPy:
    else if (py::isinstance<py::list>(data)) {
        py::list py_list = py::cast<py::list>(data);
        py::array numpy_array = py::array(py_list);
        tensor = numpy_to_torch(numpy_array);
    }
    // If it's a NumPy array:
    else if (py::isinstance<py::array>(data)) {
        py::array numpy_array = py::cast<py::array>(data);
        tensor = numpy_to_torch(numpy_array);
    }
    else {
        // Attempt to see if it's a *Python* torch.Tensor
        py::object torch_mod   = py::module_::import("torch");
        py::object torch_class = torch_mod.attr("Tensor");
        if (py::isinstance(data, torch_class)) {
            tensor = py::cast<torch::Tensor>(data);
        }
        // Or if it's already a synapx::Tensor
        else if (py::isinstance<synapx::Tensor>(data)) {
            synapx::Tensor syn_t = py::cast<synapx::Tensor>(data);
            tensor = syn_t.data();
        }
        else {
            throw std::runtime_error(
                "Data must be a scalar, Python list, NumPy array, Python torch.Tensor, or SynapX Tensor"
            );
        }
    }
    return tensor;
}

// Convert a Python torch.dtype → torch::Dtype
inline torch::Dtype pyobject_to_torch_dtype(py::object dtype_obj) {
    py::object torch_mod = py::module_::import("torch");
    
    // Check common dtypes by comparing to torch module attributes
    if (dtype_obj.is(torch_mod.attr("float32"))) {
        return torch::kFloat32;
    } else if (dtype_obj.is(torch_mod.attr("float64"))) {
        return torch::kFloat64;
    } else if (dtype_obj.is(torch_mod.attr("int32"))) {
        return torch::kInt32;
    } else if (dtype_obj.is(torch_mod.attr("int64"))) {
        return torch::kInt64;
    } else if (dtype_obj.is(torch_mod.attr("uint8"))) {
        return torch::kUInt8;
    } else if (dtype_obj.is(torch_mod.attr("int8"))) {
        return torch::kInt8;
    } else if (dtype_obj.is(torch_mod.attr("int16"))) {
        return torch::kInt16;
    } else if (dtype_obj.is(torch_mod.attr("bool"))) {
        return torch::kBool;
    } else if (dtype_obj.is(torch_mod.attr("float16"))) {
        return torch::kFloat16;
    } else {
        throw std::runtime_error("Unsupported torch dtype");
    }
}

// Convert a torch::Dtype → Python torch.dtype
inline py::object torch_dtype_to_pyobject(torch::Dtype torch_dtype) {
    py::object torch_mod = py::module_::import("torch");
    
    switch (torch_dtype) {
        case torch::kFloat32: return torch_mod.attr("float32");
        case torch::kFloat64: return torch_mod.attr("float64");
        case torch::kInt32:   return torch_mod.attr("int32");
        case torch::kInt64:   return torch_mod.attr("int64");
        case torch::kUInt8:   return torch_mod.attr("uint8");
        case torch::kInt8:    return torch_mod.attr("int8");
        case torch::kInt16:   return torch_mod.attr("int16");
        case torch::kBool:    return torch_mod.attr("bool");
        case torch::kFloat16: return torch_mod.attr("float16");
        default:
            throw std::runtime_error("Unsupported torch dtype for conversion to Python");
    }
}

inline synapx::Tensor pyobject_to_synapx(py::object obj, const synapx::Device& device) {
    if (py::isinstance<synapx::Tensor>(obj)) {
        return py::cast<synapx::Tensor>(obj);
    }
    
    // Convert scalar or other types to tensor
    torch::Tensor tensor = pyobject_to_torch(obj);
    return synapx::Tensor(tensor, false, device);
}
