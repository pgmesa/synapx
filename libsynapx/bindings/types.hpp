#ifndef BINDINGS_PYTYPES_HPP
#define BINDINGS_PYTYPES_HPP

#include <optional>

#include <synapx/synapx.hpp>

#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include "parsers.hpp"


// Context manager
class NoGradContext {
public:
    NoGradContext() : guard_(nullptr) {}
    
    void __enter__() {
        guard_ = std::make_unique<synapx::autograd::NoGradGuard>();
    }
    
    void __exit__(py::object exc_type, py::object exc_value, py::object traceback) {
        guard_.reset(); // This will call destructor and restore state
    }
    
private:
    std::unique_ptr<synapx::autograd::NoGradGuard> guard_;
};


class PyDevice {
private:
    std::optional<torch::Device> device_;

public:
    PyDevice() = default;
    
    PyDevice(const std::string& device_str) : device_(torch::Device(device_str)) {}
    
    PyDevice(const torch::Device& device) : device_(device) {}
    
    // Allow None to be passed
    PyDevice(py::object obj) {
        if (!obj.is_none()) {
            throw std::invalid_argument("Device provided is not convertible");
        }
    }
    
    bool has_value() const { return device_.has_value(); }
    
    torch::Device get() const {
        return device_.value_or(torch::kCPU);
    }

    std::string to_string() const {
        return get().str();
    }
};


class PyDtype {
private:
    std::optional<torch::Dtype> dtype_;

public:
    PyDtype() = default;
    
    PyDtype(const torch::Dtype& dtype) : dtype_(dtype) {}
    
    PyDtype(py::object obj) {
        if (!obj.is_none()) {
            throw std::invalid_argument("Device provided is not convertible");
        }
    }
    
    bool has_value() const { return dtype_.has_value(); }
    
    torch::Dtype get() const {
        return dtype_.value_or(torch::kFloat32);
    }
};


class PyTensorData {
private:
    torch::Tensor tensor_;

public:
    PyTensorData(py::object data) {
        // Handle Python scalars (int, float, bool)
        if (py::isinstance<py::int_>(data)) {
            int64_t value = py::cast<int64_t>(data);
            tensor_ = torch::tensor(value);
        }
        else if (py::isinstance<py::float_>(data)) {
            double value = py::cast<double>(data);
            tensor_ = torch::tensor(value);
        }
        else if (py::isinstance<py::bool_>(data)) {
            bool value = py::cast<bool>(data);
            tensor_ = torch::tensor(value);
        }
        // If it's a Python list, first convert to NumPy:
        else if (py::isinstance<py::list>(data)) {
            py::list py_list = py::cast<py::list>(data);
            py::array numpy_array = py::array(py_list);
            tensor_ = numpy_to_torch(numpy_array);
        }
        // If it's a NumPy array:
        else if (py::isinstance<py::array>(data)) {
            py::array numpy_array = py::cast<py::array>(data);
            tensor_ = numpy_to_torch(numpy_array);
        } 
        else {
            // Attempt to see if it's a *Python* torch.Tensor
            py::object torch_mod   = py::module_::import("torch");
            py::object torch_class = torch_mod.attr("Tensor");
            if (py::isinstance(data, torch_class)) {
                tensor_ = py::cast<torch::Tensor>(data);
            }
            // Or if it's already a synapx::Tensor
            else if (py::isinstance<synapx::Tensor>(data)) {
                throw std::runtime_error(
                    "A SynapX Tensor should not be used to create another SynapX Tensor object"
                );
            }
            else {
                throw std::runtime_error(
                    "Data must be a scalar, Python list, NumPy array or PyTorch Tensor,"
                );
            }
        }
    }
    
    torch::Tensor get() const {
        return tensor_;
    }
};


class PyDims {
private:
    std::vector<int64_t> dims_vec;

public:  
    PyDims(py::object dim) {
        if (!dim.is_none()) {
            if (py::isinstance<py::int_>(dim)) {
                int64_t dim_value = py::cast<int64_t>(dim);
                dims_vec = {dim_value};
            } else if (py::isinstance<py::iterable>(dim)) {
                for (auto item : py::cast<py::iterable>(dim)) {
                    dims_vec.push_back(py::cast<int64_t>(item));
                }
            } else {
                throw std::invalid_argument("Invalid dimension type");
            }
        }
    }
    
    std::vector<int64_t> get() const {
        return dims_vec;
    }
};

#endif