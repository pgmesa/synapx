
#include <memory>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <torch/extension.h>
#include <torch/torch.h>
#include <synapx/synapx.hpp>
#include <spdlog/spdlog.h>

#include "parsers.hpp"


namespace py = pybind11;


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

// enum class LogLevel {
//     DEBUG,
//     INFO,
//     WARNING,
//     ERROR,
//     NONE
// };

// void set_log_level(LogLevel level) {
//     switch (level) {
//         case LogLevel::DEBUG:
//             spdlog::set_level(spdlog::level::debug);
//             break;
//         case LogLevel::INFO:
//             spdlog::set_level(spdlog::level::info);
//             break;
//         case LogLevel::WARNING:
//             spdlog::set_level(spdlog::level::warn);
//             break;
//         case LogLevel::ERROR:
//             spdlog::set_level(spdlog::level::err);
//             break;
//         case LogLevel::NONE:
//             spdlog::set_level(spdlog::level::off);
//             break;
//         default:
//             throw std::invalid_argument("Unknown logging level");
//     }
// }

PYBIND11_MODULE(_C, m) {
    m.doc() = "Synapx C++ bindings";

    // Exceptions
    py::register_exception_translator([](std::exception_ptr p) {
        // Clean exceptions from libtorch 
        try {
            if (p) std::rethrow_exception(p);
        } catch (const c10::Error& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what_without_backtrace());
        } catch (const std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });

    // No grad
    py::class_<NoGradContext>(m, "NoGradContext")
        .def(py::init<>())
        .def("__enter__", &NoGradContext::__enter__)
        .def("__exit__", &NoGradContext::__exit__);

    m.def("no_grad", []() {
        return NoGradContext();
    }, "Create a context manager that disables gradient computation");
    
    m.def("is_grad_enabled", &synapx::autograd::is_grad_enabled, "Check if gradients are enabled");
    m.def("set_grad_enabled", &synapx::autograd::set_grad_enabled, "Set gradient enabled state");

    // Logging
    // py::enum_<LogLevel>(m, "LogLevel")
    //     .value("DEBUG", LogLevel::DEBUG)
    //     .value("INFO", LogLevel::INFO)
    //     .value("WARNING", LogLevel::WARNING)
    //     .value("ERROR", LogLevel::ERROR)
    //     .value("NONE", LogLevel::NONE)
    //     .export_values();

    // m.def("set_log_level", &set_log_level, "Set the current logging level.");

    // SynapX Tensor class
    py::class_<synapx::Tensor>(m, "Tensor")
        .def("numel", &synapx::Tensor::numel)
        .def("dim", &synapx::Tensor::dim)
        .def_property_readonly("shape", &synapx::Tensor::shape)
        .def("__repr__", [](const synapx::Tensor& self) -> std::string {
            py::gil_scoped_acquire gil;
            py::object py_t = py::cast(self.data());
            py::object py_repr = py::repr(py_t);
            std::string repr_str = py_repr.cast<std::string>();

            const std::string from = "tensor(";
            const std::string to = "synapx(";

            size_t pos = repr_str.find(from);
            if (pos != std::string::npos) {
                repr_str.replace(pos, from.length(), to);
            }

            return repr_str;
        })
        .def("__add__", [](const synapx::Tensor& self, py::object other) {
            synapx::Tensor other_tensor = pyobject_to_synapx(other, self.device());
            return self + other_tensor;
        }, py::arg("other"))
        .def("__mul__", [](const synapx::Tensor& self, py::object other) {
            synapx::Tensor other_tensor = pyobject_to_synapx(other, self.device());
            return self * other_tensor;
        }, py::arg("other"))
        .def("__matmul__", [](const synapx::Tensor& self, py::object other) {
            synapx::Tensor other_tensor = pyobject_to_synapx(other, self.device());
            return self.matmul(other_tensor);
        }, py::arg("other"))
        .def("__sub__", [](const synapx::Tensor& self, py::object other) {
            synapx::Tensor other_tensor = pyobject_to_synapx(other, self.device());
            return self - other_tensor;
        }, py::arg("other"))
        .def("__rsub__", [](const synapx::Tensor& self, py::object other) {
            synapx::Tensor other_tensor = pyobject_to_synapx(other, self.device());
            return other_tensor - self;
        }, py::arg("other"))
        .def("__truediv__", [](const synapx::Tensor& self, py::object other) {
            synapx::Tensor other_tensor = pyobject_to_synapx(other, self.device());
            return self / other_tensor;
        }, py::arg("other"))
        .def("__rtruediv__", [](const synapx::Tensor& self, py::object other) {
            synapx::Tensor other_tensor = pyobject_to_synapx(other, self.device());
            return other_tensor / self;
        }, py::arg("other"))
        .def("__neg__", [](const synapx::Tensor& self) {
            return -self;
        })
        .def("__radd__", [](const synapx::Tensor& self, py::object other) {
            synapx::Tensor other_tensor = pyobject_to_synapx(other, self.device());
            return other_tensor + self;
        }, py::arg("other"))
        .def("__rmul__", [](const synapx::Tensor& self, py::object other) {
            synapx::Tensor other_tensor = pyobject_to_synapx(other, self.device());
            return other_tensor * self;
        }, py::arg("other"))
        .def("__rmatmul__", [](const synapx::Tensor& self, py::object other) {
            synapx::Tensor other_tensor = pyobject_to_synapx(other, self.device());
            return other_tensor.matmul(self);
        }, py::arg("other"))
        .def("__pow__", [](const synapx::Tensor& self, py::object exponent) {
            if (py::isinstance<py::float_>(exponent) || py::isinstance<py::int_>(exponent)) {
                double exp_val = py::cast<double>(exponent);
                return self.pow(exp_val);
            } else {
                synapx::Tensor exp_tensor = pyobject_to_synapx(exponent, self.device());
                return self.pow(exp_tensor);
            }
        }, py::arg("exponent"))

        .def("__rpow__", [](const synapx::Tensor& self, py::object base) {
            synapx::Tensor base_tensor = pyobject_to_synapx(base, self.device());
            return base_tensor.pow(self);
        }, py::arg("base"))
        .def("detach", &synapx::Tensor::detach)
        .def("clone", &synapx::Tensor::clone)
        .def("add", &synapx::Tensor::add)
        .def("mul", &synapx::Tensor::mul)
        .def("matmul", &synapx::Tensor::matmul)
        .def("pow", [](const synapx::Tensor& self, py::object exponent) {
            if (py::isinstance<py::float_>(exponent) || py::isinstance<py::int_>(exponent)) {
                double exp_val = py::cast<double>(exponent);
                return self.pow(exp_val);
            } else {
                synapx::Tensor exp_tensor = pyobject_to_synapx(exponent, self.device());
                return self.pow(exp_tensor);
            }
        }, py::arg("exponent"))
        .def("exp", &synapx::Tensor::exp)
        .def("log", &synapx::Tensor::log)
        .def("sqrt", &synapx::Tensor::sqrt)
        .def("sum", [](synapx::Tensor self, py::object dim, bool keepdim = false) {
            std::vector<int64_t> dims_vec;
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
            return synapx::F::sum(self, dims_vec, keepdim);
        }, py::arg("dim") = py::none(), py::arg("keepdim") = false)
        .def_property_readonly("requires_grad", &synapx::Tensor::requires_grad)
        .def("requires_grad_", &synapx::Tensor::requires_grad_)
        .def("is_floating_point", &synapx::Tensor::is_floating_point)
        .def("item", [](const synapx::Tensor& self) -> py::object {
            return py::cast(self.data().item());
        })
        .def("zero_", &synapx::Tensor::zero_)
        .def("to", [](const synapx::Tensor& self, py::str device) {
            synapx::Device dev = string_to_device(device);
            return self.to(dev);
        }, py::arg("device"))
        .def("cpu", &synapx::Tensor::cpu)
        .def("is_leaf", &synapx::Tensor::is_leaf)
        .def("retain_grad", &synapx::Tensor::retain_grad)
        .def_property_readonly("retains_grad", &synapx::Tensor::retains_grad)
        .def_property_readonly("dtype", [](const synapx::Tensor& self) -> py::object {
            torch::Dtype torch_dtype = self.data().scalar_type();
            return torch_dtype_to_pyobject(torch_dtype);
        }, "Underlying torch dtype")
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
            } else if (py::isinstance<torch::Tensor>(grad)) {
                torch::Tensor grad_tensor = py::cast<torch::Tensor>(grad);
                self.backward(grad_tensor);
            } else {
                throw std::runtime_error("grad must be a synapx::Tensor or torch::Tensor");
            }
        }, py::arg("grad") = py::none(), 
           "Union[None, synapx.Tensor, torch.Tensor]: Computes the gradient of current tensor w.r.t. graph leaves.")
        .def_property_readonly("data", &synapx::Tensor::data)
        .def("torch", &synapx::Tensor::data)
        .def("numpy", [](const synapx::Tensor& self) {
            return torch_to_numpy(self.data());
        });

    m.def("tensor", [](py::object data, bool requires_grad, std::string device, py::object dtype) {
        synapx::Device dev = string_to_device(device);
        torch::Tensor tensor = pyobject_to_torch(data);
        
        // Convert dtype if specified
        if (!dtype.is_none()) {
            torch::Dtype torch_dtype = pyobject_to_torch_dtype(dtype);
            tensor = tensor.to(torch_dtype);
        }
        
        return synapx::Tensor(tensor, requires_grad, dev);
    }, py::arg("data"), py::arg("requires_grad") = false, py::arg("device") = "cpu", py::arg("dtype") = py::none());

    m.def("ones", [](py::iterable shape, bool requires_grad, std::string device, py::object dtype) {
        synapx::Device dev = string_to_device(device);
        std::vector<int64_t> dims;
        for (auto item : shape) {
            dims.push_back(py::cast<int64_t>(item));
        }
        
        torch::TensorOptions options = torch::TensorOptions();
        if (!dtype.is_none()) {
            torch::Dtype torch_dtype = pyobject_to_torch_dtype(dtype);
            options = options.dtype(torch_dtype);
        }
        
        return synapx::Tensor(torch::ones(dims, options), requires_grad, dev);
    }, py::arg("shape"), py::arg("requires_grad") = false, py::arg("device") = "cpu", py::arg("dtype") = py::none());

    m.def("zeros", [](py::iterable shape, bool requires_grad, std::string device, py::object dtype) {
        synapx::Device dev = string_to_device(device);
        std::vector<int64_t> dims;
        for (auto item : shape) {
            dims.push_back(py::cast<int64_t>(item));
        }
        
        torch::TensorOptions options = torch::TensorOptions();
        if (!dtype.is_none()) {
            torch::Dtype torch_dtype = pyobject_to_torch_dtype(dtype);
            options = options.dtype(torch_dtype);
        }
        
        return synapx::Tensor(torch::zeros(dims, options), requires_grad, dev);
    }, py::arg("shape"), py::arg("requires_grad") = false, py::arg("device") = "cpu", py::arg("dtype") = py::none());

    m.def("ones_like", [](const synapx::Tensor& input, bool requires_grad, std::string device, py::object dtype) {
        synapx::Device dev = string_to_device(device);
        
        torch::TensorOptions options = torch::TensorOptions();
        if (!dtype.is_none()) {
            torch::Dtype torch_dtype = pyobject_to_torch_dtype(dtype);
            options = options.dtype(torch_dtype);
        }
        
        torch::Tensor result = torch::ones_like(input.data(), options);
        return synapx::Tensor(result, requires_grad, dev);
    }, py::arg("input"), py::arg("requires_grad") = false, py::arg("device") = "cpu", py::arg("dtype") = py::none());

    m.def("zeros_like", [](const synapx::Tensor& input, bool requires_grad, std::string device, py::object dtype) {
        synapx::Device dev = string_to_device(device);
        
        torch::TensorOptions options = torch::TensorOptions();
        if (!dtype.is_none()) {
            torch::Dtype torch_dtype = pyobject_to_torch_dtype(dtype);
            options = options.dtype(torch_dtype);
        }
        
        torch::Tensor result = torch::zeros_like(input.data(), options);
        return synapx::Tensor(result, requires_grad, dev);
    }, py::arg("input"), py::arg("requires_grad") = false, py::arg("device") = "cpu", py::arg("dtype") = py::none());
        
    m.def("add", [](synapx::Tensor t1, synapx::Tensor t2) {
        return synapx::F::add(t1, t2);
    }, py::arg("t1"), py::arg("t2"));
    
    m.def("mul", [](synapx::Tensor t1, synapx::Tensor t2) {
        return synapx::F::mul(t1, t2);
    }, py::arg("t1"), py::arg("t2"));
    
    m.def("matmul", [](synapx::Tensor t1, synapx::Tensor t2) {
        return synapx::F::matmul(t1, t2);
    }, py::arg("t1"), py::arg("t2"));

    m.def("addmm", [](synapx::Tensor inp, synapx::Tensor mat1, synapx::Tensor mat2) {
        return synapx::F::addmm(inp, mat1, mat2);
    }, py::arg("inp"), py::arg("mat1"), py::arg("mat2"));

    m.def("pow", [](synapx::Tensor t1, py::object exponent) {
        if (py::isinstance<py::float_>(exponent) || py::isinstance<py::int_>(exponent)) {
            double exp_val = py::cast<double>(exponent);
            return synapx::F::pow(t1, exp_val);
        } else {
            synapx::Tensor exp_tensor = pyobject_to_synapx(exponent, t1.device());
            return synapx::F::pow(t1, exp_tensor);
        }
    }, py::arg("t1"), py::arg("exponent"));

    m.def("clone", [](synapx::Tensor t1) {
        return synapx::F::clone(t1);  
    }, py::arg("t1"));

    m.def("exp", [](synapx::Tensor t1) {
        return synapx::F::exp(t1);  
    }, py::arg("t1"));

    m.def("log", [](synapx::Tensor t1) {
        return synapx::F::log(t1);  
    }, py::arg("t1"));

    m.def("sqrt", [](synapx::Tensor t1) {
        return synapx::F::sqrt(t1);  
    }, py::arg("t1"));

    m.def("sum", [](synapx::Tensor t1, py::object dim, bool keepdim = false) {
        std::vector<int64_t> dims_vec;
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
        return synapx::F::sum(t1, dims_vec, keepdim);
    }, py::arg("t1"), py::arg("dim") = py::none(), py::arg("keepdim") = false);

}
