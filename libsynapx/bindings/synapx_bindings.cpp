
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
        .def_property_readonly("data", &synapx::Tensor::data)
        .def_property_readonly("shape", &synapx::Tensor::shape)
        .def_property_readonly("requires_grad", &synapx::Tensor::requires_grad)
        .def_property_readonly("retains_grad", &synapx::Tensor::retains_grad)
        .def_property_readonly("is_leaf", &synapx::Tensor::is_leaf)
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
        .def("numel", &synapx::Tensor::numel)
        .def("dim", &synapx::Tensor::dim)
        .def("requires_grad_", &synapx::Tensor::requires_grad_)
        .def("is_floating_point", &synapx::Tensor::is_floating_point)
        .def("retain_grad", &synapx::Tensor::retain_grad)
        .def("item", [](const synapx::Tensor& self) -> py::object {
            return py::cast(self.data().item());
        })
        .def("to", [](const synapx::Tensor& self, py::str device) -> synapx::Tensor {
            synapx::Device dev = string_to_device(device);
            return self.to(dev);
        }, py::arg("device"))
        .def("cpu", &synapx::Tensor::cpu)
        .def("torch", &synapx::Tensor::data)
        .def("numpy", [](const synapx::Tensor& self) -> py::array {
            return torch_to_numpy(self.data());
        })
        .def("detach", &synapx::Tensor::detach)
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

        // Normal basic operations
        .def("__add__", py::overload_cast<const synapx::Tensor&>(&synapx::Tensor::add, py::const_), py::arg("other"))
        .def("__add__", py::overload_cast<double>(&synapx::Tensor::add, py::const_), py::arg("other"))

        .def("__sub__", py::overload_cast<const synapx::Tensor&>(&synapx::Tensor::sub, py::const_), py::arg("other"))
        .def("__sub__", py::overload_cast<double>(&synapx::Tensor::sub, py::const_), py::arg("other"))

        .def("__mul__", py::overload_cast<const synapx::Tensor&>(&synapx::Tensor::mul, py::const_), py::arg("other"))
        .def("__mul__", py::overload_cast<double>(&synapx::Tensor::mul, py::const_), py::arg("other"))

        .def("__truediv__", py::overload_cast<const synapx::Tensor&>(&synapx::Tensor::div, py::const_), py::arg("other"))
        .def("__truediv__", py::overload_cast<double>(&synapx::Tensor::div, py::const_), py::arg("other"))

        .def("__matmul__", &synapx::Tensor::matmul, py::arg("other"))

        .def("__pow__", py::overload_cast<const synapx::Tensor&>(&synapx::Tensor::pow, py::const_), py::arg("exponent"))
        .def("__pow__", py::overload_cast<double>(&synapx::Tensor::pow, py::const_), py::arg("exponent"))

        .def("__neg__", &synapx::Tensor::neg)


        // In-place operations
        .def("__iadd__", py::overload_cast<const synapx::Tensor&>(&synapx::Tensor::add_), py::arg("other"))
        .def("__iadd__", py::overload_cast<double>(&synapx::Tensor::add_), py::arg("other"))

        .def("__isub__", py::overload_cast<const synapx::Tensor&>(&synapx::Tensor::sub_), py::arg("other"))
        .def("__isub__", py::overload_cast<double>(&synapx::Tensor::sub_), py::arg("other"))

        .def("__imul__", py::overload_cast<const synapx::Tensor&>(&synapx::Tensor::mul_), py::arg("other"))
        .def("__imul__", py::overload_cast<double>(&synapx::Tensor::mul_), py::arg("other"))

        .def("__itruediv__", py::overload_cast<const synapx::Tensor&>(&synapx::Tensor::div_), py::arg("other"))
        .def("__itruediv__", py::overload_cast<double>(&synapx::Tensor::div_), py::arg("other"))


        // Reverse operations
        .def("__radd__", py::overload_cast<const synapx::Tensor&>(&synapx::Tensor::add, py::const_), py::arg("other"))
        .def("__radd__", py::overload_cast<double>(&synapx::Tensor::add, py::const_), py::arg("other"))

        .def("__rsub__", py::overload_cast<const synapx::Tensor&>(&synapx::Tensor::rsub, py::const_), py::arg("other"))
        .def("__rsub__", py::overload_cast<double>(&synapx::Tensor::rsub, py::const_), py::arg("other"))

        .def("__rmul__", py::overload_cast<const synapx::Tensor&>(&synapx::Tensor::mul, py::const_), py::arg("other"))
        .def("__rmul__", py::overload_cast<double>(&synapx::Tensor::mul, py::const_), py::arg("other"))

        .def("__rtruediv__", py::overload_cast<const synapx::Tensor&>(&synapx::Tensor::rdiv, py::const_), py::arg("other"))
        .def("__rtruediv__", py::overload_cast<double>(&synapx::Tensor::rdiv, py::const_), py::arg("other"))

        .def("__rmatmul__", &synapx::Tensor::rmatmul, py::arg("other"))

        .def("__rpow__", py::overload_cast<const synapx::Tensor&>(&synapx::Tensor::rpow, py::const_), py::arg("base"))
        .def("__rpow__", py::overload_cast<double>(&synapx::Tensor::rpow, py::const_), py::arg("base"))


        // Functions to access operations
        .def("add", py::overload_cast<const synapx::Tensor&>(&synapx::Tensor::add, py::const_), py::arg("other"))
        .def("add", py::overload_cast<double>(&synapx::Tensor::add, py::const_), py::arg("scalar"))

        .def("sub", py::overload_cast<const synapx::Tensor&>(&synapx::Tensor::sub, py::const_), py::arg("other"))
        .def("sub", py::overload_cast<double>(&synapx::Tensor::sub, py::const_), py::arg("other"))

        .def("mul", py::overload_cast<const synapx::Tensor&>(&synapx::Tensor::mul, py::const_), py::arg("other"))
        .def("mul", py::overload_cast<double>(&synapx::Tensor::mul, py::const_), py::arg("other"))

        .def("pow", py::overload_cast<const synapx::Tensor&>(&synapx::Tensor::pow, py::const_), py::arg("exponent"))
        .def("pow", py::overload_cast<double>(&synapx::Tensor::pow, py::const_), py::arg("exponent"))

        .def("div", py::overload_cast<const synapx::Tensor&>(&synapx::Tensor::div, py::const_), py::arg("other"))
        .def("div", py::overload_cast<double>(&synapx::Tensor::div, py::const_), py::arg("other"))

        .def("matmul", &synapx::Tensor::matmul, py::arg("other"))

        .def("neg", &synapx::Tensor::neg)


        // In-place
        .def("add_", py::overload_cast<const synapx::Tensor&>(&synapx::Tensor::add_), py::arg("other"))
        .def("add_", py::overload_cast<double>(&synapx::Tensor::add_), py::arg("other"))

        .def("sub_", py::overload_cast<const synapx::Tensor&>(&synapx::Tensor::sub_), py::arg("other"))
        .def("sub_", py::overload_cast<double>(&synapx::Tensor::sub_), py::arg("other"))

        .def("mul_", py::overload_cast<const synapx::Tensor&>(&synapx::Tensor::mul_), py::arg("other"))
        .def("mul_", py::overload_cast<double>(&synapx::Tensor::mul_), py::arg("other"))

        .def("pow_", py::overload_cast<const synapx::Tensor&>(&synapx::Tensor::pow_), py::arg("other"))
        .def("pow_", py::overload_cast<double>(&synapx::Tensor::pow_), py::arg("other"))

        .def("div_", py::overload_cast<const synapx::Tensor&>(&synapx::Tensor::div_), py::arg("other"))
        .def("div_", py::overload_cast<double>(&synapx::Tensor::div_), py::arg("other"))
        
        .def("neg_", &synapx::Tensor::neg_)
        .def("zero_", &synapx::Tensor::zero_)
        
        // Other operations
        .def("clone", &synapx::Tensor::clone)

        .def("exp", &synapx::Tensor::exp)

        .def("log", &synapx::Tensor::log)

        .def("sqrt", &synapx::Tensor::sqrt)

        .def("sum", [](synapx::Tensor self, py::object dim, bool keepdim = false) -> synapx::Tensor {
            std::vector<int64_t> dims_vec = pyobj_to_dims(dim);
            return self.sum(dims_vec, keepdim);
        }, py::arg("dim") = py::none(), py::arg("keepdim") = false)

        .def("mean", [](synapx::Tensor self, py::object dim, bool keepdim = false) -> synapx::Tensor {
            std::vector<int64_t> dims_vec = pyobj_to_dims(dim);
            return self.mean(dims_vec, keepdim);
        }, py::arg("dim") = py::none(), py::arg("keepdim") = false)
        
        .def("max", [](const synapx::Tensor& self) { 
            return self.max(); 
        })
        .def("max", [](const synapx::Tensor& self, int64_t dim, bool keepdim) { 
            return self.max(dim, keepdim); 
        }, py::arg("dim"), py::arg("keepdim") = false)

        .def("min", [](const synapx::Tensor& self) { 
            return self.min(); 
        })
        .def("min", [](const synapx::Tensor& self, int64_t dim, bool keepdim) { 
            return self.min(dim, keepdim); 
        }, py::arg("dim"), py::arg("keepdim") = false)
        ;
    
    // Initializers
    m.def("tensor", [](py::object data, bool requires_grad, std::string device, py::object dtype) -> synapx::Tensor {
        synapx::Device dev = string_to_device(device);
        torch::Tensor tensor = pyobj_to_torch(data);
        
        // Convert dtype if specified
        torch::Dtype torch_dtype = torch::kFloat32;
        if (!dtype.is_none()) {
            torch_dtype = pyobj_to_torch_dtype(dtype);
        }
        tensor = tensor.to(torch_dtype);
        
        return synapx::Tensor(tensor, requires_grad, dev);
    }, py::arg("data"), py::arg("requires_grad") = false, py::arg("device") = "cpu", py::arg("dtype") = py::none());

    m.def("ones", [](py::object shape, bool requires_grad, std::string device, py::object dtype) -> synapx::Tensor {
        synapx::Device dev = string_to_device(device);
        std::vector<int64_t> dims = pyobj_to_dims(shape);
        
        torch::TensorOptions options = torch::TensorOptions();
        torch::Dtype torch_dtype = torch::kFloat32;
        if (!dtype.is_none()) {
            torch_dtype = pyobj_to_torch_dtype(dtype);
        }
        options = options.dtype(torch_dtype);
        
        return synapx::Tensor(torch::ones(dims, options), requires_grad, dev);
    }, py::arg("shape"), py::arg("requires_grad") = false, py::arg("device") = "cpu", py::arg("dtype") = py::none());

    m.def("ones_like", [](const synapx::Tensor& input, bool requires_grad, std::string device, py::object dtype) -> synapx::Tensor {
        synapx::Device dev = string_to_device(device);
        
        torch::TensorOptions options = input.options();
        if (!dtype.is_none()) {
            torch::Dtype torch_dtype = pyobj_to_torch_dtype(dtype);
            options = options.dtype(torch_dtype);
        }
        
        torch::Tensor result = torch::ones_like(input.data(), options);
        return synapx::Tensor(result, requires_grad, dev);
    }, py::arg("input"), py::arg("requires_grad") = false, py::arg("device") = "cpu", py::arg("dtype") = py::none());

    m.def("zeros", [](py::object shape, bool requires_grad, std::string device, py::object dtype) -> synapx::Tensor {
        synapx::Device dev = string_to_device(device);
        std::vector<int64_t> dims = pyobj_to_dims(shape);
        
        torch::TensorOptions options = torch::TensorOptions();
        torch::Dtype torch_dtype = torch::kFloat32;
        if (!dtype.is_none()) {
            torch_dtype = pyobj_to_torch_dtype(dtype);
        }
        options = options.dtype(torch_dtype);
        
        return synapx::Tensor(torch::zeros(dims, options), requires_grad, dev);
    }, py::arg("shape"), py::arg("requires_grad") = false, py::arg("device") = "cpu", py::arg("dtype") = py::none());

    m.def("zeros_like", [](const synapx::Tensor& input, bool requires_grad, std::string device, py::object dtype) -> synapx::Tensor {
        synapx::Device dev = string_to_device(device);
        
        torch::TensorOptions options = input.options();
        if (!dtype.is_none()) {
            torch::Dtype torch_dtype = pyobj_to_torch_dtype(dtype);
            options = options.dtype(torch_dtype);
        }
        
        torch::Tensor result = torch::zeros_like(input.data(), options);
        return synapx::Tensor(result, requires_grad, dev);
    }, py::arg("input"), py::arg("requires_grad") = false, py::arg("device") = "cpu", py::arg("dtype") = py::none());

    m.def("rand", [](py::object shape, bool requires_grad, std::string device, py::object dtype) -> synapx::Tensor {
        synapx::Device dev = string_to_device(device);
        std::vector<int64_t> dims = pyobj_to_dims(shape);
        
        torch::TensorOptions options = torch::TensorOptions();
        torch::Dtype torch_dtype = torch::kFloat32;
        if (!dtype.is_none()) {
            torch_dtype = pyobj_to_torch_dtype(dtype);
        }
        options = options.dtype(torch_dtype);
        
        return synapx::Tensor(torch::rand(dims, options), requires_grad, dev);
    }, py::arg("shape"), py::arg("requires_grad") = false, py::arg("device") = "cpu", py::arg("dtype") = py::none());

    m.def("rand_like", [](const synapx::Tensor& input, bool requires_grad, std::string device, py::object dtype) -> synapx::Tensor {
        synapx::Device dev = string_to_device(device);
        
        torch::TensorOptions options = input.options();
        if (!dtype.is_none()) {
            torch::Dtype torch_dtype = pyobj_to_torch_dtype(dtype);
            options = options.dtype(torch_dtype);
        }
        
        torch::Tensor result = torch::rand_like(input.data(), options);
        return synapx::Tensor(result, requires_grad, dev);
    }, py::arg("input"), py::arg("requires_grad") = false, py::arg("device") = "cpu", py::arg("dtype") = py::none());
       
    // Basic operations
    m.def("add", py::overload_cast<const synapx::Tensor&, const synapx::Tensor&>(&synapx::add), py::arg("t1"), py::arg("t2"));
    m.def("add", py::overload_cast<const synapx::Tensor&, double>(&synapx::add), py::arg("tensor"), py::arg("scalar"));
    
    m.def("sub", py::overload_cast<const synapx::Tensor&, const synapx::Tensor&>(&synapx::sub), py::arg("t1"), py::arg("t2"));
    m.def("sub", py::overload_cast<const synapx::Tensor&, double>(&synapx::sub), py::arg("tensor"), py::arg("scalar"));
    
    m.def("mul", py::overload_cast<const synapx::Tensor&, const synapx::Tensor&>(&synapx::mul), py::arg("t1"), py::arg("t2"));
    m.def("mul", py::overload_cast<const synapx::Tensor&, double>(&synapx::mul), py::arg("tensor"), py::arg("scalar"));
    
    m.def("pow", py::overload_cast<const synapx::Tensor&, const synapx::Tensor&>(&synapx::pow), py::arg("tensor"), py::arg("exponent"));
    m.def("pow", py::overload_cast<const synapx::Tensor&, double>(&synapx::pow), py::arg("tensor"), py::arg("exponent"));
    
    m.def("div", py::overload_cast<const synapx::Tensor&, const synapx::Tensor&>(&synapx::div), py::arg("t1"), py::arg("t2"));
    m.def("div", py::overload_cast<const synapx::Tensor&, double>(&synapx::div), py::arg("tensor"), py::arg("scalar"));
    
    m.def("matmul", &synapx::matmul, py::arg("t1"), py::arg("t2"));
    
    m.def("neg", &synapx::neg);

    // Other operations
    m.def("addmm", &synapx::addmm, py::arg("inp"), py::arg("mat1"), py::arg("mat2"));

    m.def("clone", synapx::clone, py::arg("tensor"));

    m.def("exp", synapx::exp, py::arg("tensor"));

    m.def("log", synapx::log, py::arg("tensor"));

    m.def("sqrt", synapx::sqrt, py::arg("tensor"));

    m.def("sum", [](synapx::Tensor tensor, py::object dim, bool keepdim = false) -> synapx::Tensor {
        std::vector<int64_t> dims_vec = pyobj_to_dims(dim);
        return synapx::sum(tensor, dims_vec, keepdim);
    }, py::arg("tensor"), py::arg("dim") = py::none(), py::arg("keepdim") = false);

    m.def("mean", [](synapx::Tensor tensor, py::object dim, bool keepdim = false) -> synapx::Tensor {
        std::vector<int64_t> dims_vec = pyobj_to_dims(dim);
        return synapx::mean(tensor, dims_vec, keepdim);
    }, py::arg("tensor"), py::arg("dim") = py::none(), py::arg("keepdim") = false);

    m.def("max", [](const synapx::Tensor& tensor) { return synapx::max(tensor); }, py::arg("tensor"));
    m.def("max", [](const synapx::Tensor& tensor, int64_t dim, bool keepdim) { 
        return synapx::max(tensor, dim, keepdim); 
    }, py::arg("tensor"), py::arg("dim"), py::arg("keepdim") = false);

    m.def("min", [](const synapx::Tensor& tensor) { return synapx::min(tensor); }, py::arg("tensor"));
    m.def("min", [](const synapx::Tensor& tensor, int64_t dim, bool keepdim) { 
        return synapx::min(tensor, dim, keepdim); 
    }, py::arg("tensor"), py::arg("dim"), py::arg("keepdim") = false);

}
