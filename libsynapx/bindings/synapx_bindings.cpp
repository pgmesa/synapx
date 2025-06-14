
#include <memory>
#include <optional>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <torch/extension.h>
#include <torch/torch.h>

#include <synapx/synapx.hpp>

#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include "parsers.hpp"
#include "types.hpp"
#include "config.hpp"


namespace py = pybind11;


bool TensorReprConfig::detailed_repr = false;
LogLevel LoggingConfig::debug_level = LogLevel::INFO_LEVEL;


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

    // Device
    py::class_<PyDevice>(m, "device")
        .def(py::init<>())
        .def(py::init<const std::string&>())
        .def(py::init<torch::Device>())
        .def(py::init<py::object>())
        .def("torch", &PyDevice::get)
        .def("__str__", &PyDevice::to_string)
        .def("__repr__", [](const PyDevice& d) { return "device('" + d.to_string() + "')"; });
        ;

    py::implicitly_convertible<py::str, PyDevice>();
    py::implicitly_convertible<py::object, PyDevice>();
    py::implicitly_convertible<torch::Device, PyDevice>();

    // Dtype
    py::class_<PyDtype>(m, "dtype")
        .def(py::init<>())
        .def(py::init<torch::Dtype>())
        .def(py::init<py::object>())
        .def("torch", &PyDtype::get)
        ;

    py::implicitly_convertible<py::object, PyDtype>();
    py::implicitly_convertible<torch::Device, PyDtype>();

    // Dimensions
    py::class_<PyDims>(m, "dims")
        .def(py::init<py::object>())
        .def("tolist", &PyDims::get)
        ;

    py::implicitly_convertible<py::object, PyDims>();

    // Tensor Data
    py::class_<PyTensorData>(m, "TensorDataContainer")
        .def(py::init<py::object>())
        .def("torch", &PyTensorData::get)
        ;

    py::implicitly_convertible<py::object, PyTensorData>();

    // LoggingConfig
    pybind11::enum_<LogLevel>(m, "LogLevel")
        .value("DEBUG", LogLevel::DEBUG_LEVEL)
        .value("INFO", LogLevel::INFO_LEVEL)
        .value("WARNING", LogLevel::WARNING_LEVEL)
        .value("ERROR", LogLevel::ERROR_LEVEL)
        .value("NONE", LogLevel::NONE_LEVEL);

    m.def("set_log_level", &LoggingConfig::set_log_level,
           py::arg("level"), "Set the logging level");
    m.def("get_log_level", &LoggingConfig::get_log_level,
          "Get the current logging level");

    // Tensor representation config
    m.def("set_detailed_repr", &TensorReprConfig::set_detailed_repr, 
          py::arg("flag"), "Enable or disable detailed tensor representation");
    m.def("is_detailed_repr_enabled", &TensorReprConfig::is_detailed_repr_enabled, 
          "Check if detailed tensor representation is enabled");

    // SynapX Tensor class
    py::class_<synapx::Tensor>(m, "Tensor")
        .def_property_readonly("data", &synapx::Tensor::data)
        .def_property_readonly("shape", [](const synapx::Tensor& self) -> py::tuple {
            pybind11::tuple py_shape = pybind11::cast(self.shape());
            return py_shape;
        })
        .def_property_readonly("requires_grad", &synapx::Tensor::requires_grad)
        .def_property_readonly("retains_grad", &synapx::Tensor::retains_grad)
        .def_property_readonly("is_leaf", &synapx::Tensor::is_leaf)
        .def_property_readonly("dtype", &synapx::Tensor::dtype, "Underlying torch dtype")
        .def_property_readonly("device", &synapx::Tensor::device, "Underlying torch device")
        .def_property_readonly("grad", [](const synapx::Tensor& self) -> py::object {
            synapx::Tensor grad_tensor = self.grad();
            if (grad_tensor.defined()) {
                return py::cast(grad_tensor);
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
        .def("to", [](const synapx::Tensor& self, PyDevice device) -> synapx::Tensor {
            return self.to(device.get());
        }, py::arg("device"))
        .def("to", [](const synapx::Tensor& self, PyDtype dtype) -> synapx::Tensor {
            return self.to(dtype.get());
        }, py::arg("dtype"))
        .def("cpu", &synapx::Tensor::cpu)
        .def("cuda", &synapx::Tensor::cuda, py::arg("index") = 0)
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
            
            if (self.grad_fn()) {
                std::string tensor_info = fmt::format(", grad_fn=<{}>)", self.grad_fn()->name());
                const std::string end = ")";
                size_t end_pos = repr_str.find(end);
                if (end_pos != std::string::npos) {
                    repr_str.replace(end_pos, end.length(), tensor_info);
                }
            }

            return repr_str;
        })

        .def("__len__", [](const synapx::Tensor& self) {
            return self.shape()[0];
        })

        .def("__getitem__", [](const synapx::Tensor& self, const py::object& key) {
            auto indices = TensorIndexConverter::convert(key, self);
            return self[indices];
        }, py::arg("key"), "Index tensor using Python notation")

        // Add basic setitem functionality (if more advanced is required the underlying torch data can be used)
        .def("__setitem__", [](synapx::Tensor& self, const py::object& key, const synapx::Tensor& value) {
            auto indices = TensorIndexConverter::convert(key, self);
            self.index_put_(indices, value);
        }, py::arg("indices"), py::arg("value"))

        .def("__setitem__", [](synapx::Tensor& self, const py::object& key, double value) {
            auto indices = TensorIndexConverter::convert(key, self);
            self.index_put_(indices, value);
        }, py::arg("indices"), py::arg("value"))

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

        .def("sum", [](synapx::Tensor self, PyDims dim, bool keepdim = false) -> synapx::Tensor {
            return self.sum(dim.get(), keepdim);
        }, py::arg("dim") = py::none(), py::arg("keepdim") = false)

        .def("mean", [](synapx::Tensor self, PyDims dim, bool keepdim = false) -> synapx::Tensor {
            return self.mean(dim.get(), keepdim);
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

        .def("squeeze", [](synapx::Tensor self, PyDims dim) -> synapx::Tensor {
            return self.squeeze(dim.get());
        }, py::arg("dim") = py::none())

        .def("unsqueeze", &synapx::Tensor::unsqueeze, py::arg("dim"))
        
        .def("reshape", &synapx::Tensor::reshape, py::arg("shape"))

        .def("transpose", &synapx::Tensor::transpose, py::arg("dim0"), py::arg("dim1"))

        .def("swapdims", &synapx::Tensor::swapdims, py::arg("dim0"), py::arg("dim1"))
        
        .def("movedim", &synapx::Tensor::movedim, py::arg("source"), py::arg("destination"))
        ;
    
    // Initializers
    m.def("tensor", [](PyTensorData data, bool requires_grad, PyDevice device, PyDtype dtype) {
        torch::Tensor tensor = data.get().to(device.get()).to(dtype.get());
        return synapx::Tensor(tensor, requires_grad);
    }, py::arg("data"), py::arg("requires_grad") = false, py::arg("device") = py::none(), py::arg("dtype") = py::none());


    m.def("ones", [](PyDims shape, bool requires_grad, PyDevice device, PyDtype dtype) {
        torch::TensorOptions options = torch::TensorOptions();
        options = options.device(device.get()).dtype(dtype.get());
        
        return synapx::ones(shape.get(), options, requires_grad);
    }, py::arg("shape"), py::arg("requires_grad") = false, py::arg("device") = py::none(), py::arg("dtype") = py::none());

    m.def("ones_like", [](const synapx::Tensor& input, bool requires_grad, PyDevice device, PyDtype dtype) {
        torch::TensorOptions options = input.options();
        if (device.has_value()) options = options.device(device.get());
        if (dtype.has_value()) options = options.dtype(dtype.get());
        
        return synapx::ones_like(input, options, requires_grad);
    }, py::arg("input"), py::arg("requires_grad") = false, py::arg("device") = py::none(), py::arg("dtype") = py::none());


    m.def("zeros", [](PyDims shape, bool requires_grad, PyDevice device, PyDtype dtype) {
        torch::TensorOptions options = torch::TensorOptions();
        options = options.device(device.get()).dtype(dtype.get());
        
        return synapx::zeros(shape.get(), options, requires_grad);
    }, py::arg("shape"), py::arg("requires_grad") = false, py::arg("device") = py::none(), py::arg("dtype") = py::none());

    m.def("zeros_like", [](const synapx::Tensor& input, bool requires_grad, PyDevice device, PyDtype dtype) {
        torch::TensorOptions options = input.options();
        if (device.has_value()) options = options.device(device.get());
        if (dtype.has_value()) options = options.dtype(dtype.get());
        
        return synapx::zeros_like(input, options, requires_grad);
    }, py::arg("input"), py::arg("requires_grad") = false, py::arg("device") = py::none(), py::arg("dtype") = py::none());
    

    m.def("rand", [](PyDims shape, bool requires_grad, PyDevice device, PyDtype dtype) {
        torch::TensorOptions options = torch::TensorOptions();
        options = options.device(device.get()).dtype(dtype.get());
        
        return synapx::rand(shape.get(), options, requires_grad);
    }, py::arg("shape"), py::arg("requires_grad") = false, py::arg("device") = py::none(), py::arg("dtype") = py::none());

    m.def("rand_like", [](const synapx::Tensor& input, bool requires_grad, PyDevice device, PyDtype dtype) {
        torch::TensorOptions options = input.options();
        if (device.has_value()) options = options.device(device.get());
        if (dtype.has_value()) options = options.dtype(dtype.get());
        
        return synapx::rand_like(input, options, requires_grad);
    }, py::arg("input"), py::arg("requires_grad") = false, py::arg("device") = py::none(), py::arg("dtype") = py::none());


    m.def("randn", [](PyDims shape, bool requires_grad, PyDevice device, PyDtype dtype) {
        torch::TensorOptions options = torch::TensorOptions();
        options = options.device(device.get()).dtype(dtype.get());
        
        return synapx::randn(shape.get(), options, requires_grad);
    }, py::arg("shape"), py::arg("requires_grad") = false, py::arg("device") = py::none(), py::arg("dtype") = py::none());

    m.def("randn_like", [](const synapx::Tensor& input, bool requires_grad, PyDevice device, PyDtype dtype) {
        torch::TensorOptions options = input.options();
        if (device.has_value()) options = options.device(device.get());
        if (dtype.has_value()) options = options.dtype(dtype.get());
        
        return synapx::randn_like(input, options, requires_grad);
    }, py::arg("input"), py::arg("requires_grad") = false, py::arg("device") = py::none(), py::arg("dtype") = py::none());


    m.def("empty", [](PyDims shape, bool requires_grad, PyDevice device, PyDtype dtype) {
        torch::TensorOptions options = torch::TensorOptions();
        options = options.device(device.get()).dtype(dtype.get());
        
        return synapx::empty(shape.get(), options, requires_grad);
    }, py::arg("shape"), py::arg("requires_grad") = false, py::arg("device") = py::none(), py::arg("dtype") = py::none());

    m.def("empty_like", [](const synapx::Tensor& input, bool requires_grad, PyDevice device, PyDtype dtype) {
        torch::TensorOptions options = input.options();
        if (device.has_value()) options = options.device(device.get());
        if (dtype.has_value()) options = options.dtype(dtype.get());
        
        return synapx::empty_like(input, options, requires_grad);
    }, py::arg("input"), py::arg("requires_grad") = false, py::arg("device") = py::none(), py::arg("dtype") = py::none());


    m.def("full", [](PyDims shape, double fill_value, bool requires_grad, PyDevice device, PyDtype dtype) {
        torch::TensorOptions options = torch::TensorOptions();
        options = options.device(device.get()).dtype(dtype.get());
        
        return synapx::full(shape.get(), fill_value, options, requires_grad);
    }, py::arg("shape"), py::arg("fill_value"), py::arg("requires_grad") = false, py::arg("device") = py::none(), py::arg("dtype") = py::none());

    m.def("full_like", [](const synapx::Tensor& input, double fill_value, bool requires_grad, PyDevice device, PyDtype dtype) {
        torch::TensorOptions options = input.options();
        if (device.has_value()) options = options.device(device.get());
        if (dtype.has_value()) options = options.dtype(dtype.get());
        
        return synapx::full_like(input, fill_value, options, requires_grad);
    }, py::arg("input"), py::arg("fill_value"), py::arg("requires_grad") = false, py::arg("device") = py::none(), py::arg("dtype") = py::none());
       

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

    m.def("clone", &synapx::clone, py::arg("tensor"));

    m.def("exp", &synapx::exp, py::arg("tensor"));

    m.def("log", &synapx::log, py::arg("tensor"));

    m.def("sqrt", &synapx::sqrt, py::arg("tensor"));

    m.def("sum", [](synapx::Tensor tensor, PyDims dim, bool keepdim = false) {
        return synapx::sum(tensor, dim.get(), keepdim);
    }, py::arg("tensor"), py::arg("dim") = py::none(), py::arg("keepdim") = false);

    m.def("mean", [](synapx::Tensor tensor, PyDims dim, bool keepdim = false) {
        return synapx::mean(tensor, dim.get(), keepdim);
    }, py::arg("tensor"), py::arg("dim") = py::none(), py::arg("keepdim") = false);

    m.def("max", [](const synapx::Tensor& tensor) { 
        return synapx::max(tensor); 
    }, py::arg("tensor"));
    m.def("max", [](const synapx::Tensor& tensor, int64_t dim, bool keepdim) { 
        return synapx::max(tensor, dim, keepdim); 
    }, py::arg("tensor"), py::arg("dim"), py::arg("keepdim") = false);

    m.def("min", [](const synapx::Tensor& tensor) { 
        return synapx::min(tensor); 
    }, py::arg("tensor"));
    m.def("min", [](const synapx::Tensor& tensor, int64_t dim, bool keepdim) { 
        return synapx::min(tensor, dim, keepdim); 
    }, py::arg("tensor"), py::arg("dim"), py::arg("keepdim") = false);

    m.def("squeeze", [](synapx::Tensor tensor, PyDims dim) {
        return synapx::squeeze(tensor, dim.get());
    }, py::arg("tensor"), py::arg("dim") = py::none());

    m.def("unsqueeze", &synapx::unsqueeze, py::arg("tensor"), py::arg("dim"));

    m.def("reshape", &synapx::reshape, py::arg("tensor"), py::arg("shape"));

    m.def("transpose", &synapx::transpose, py::arg("tensor"), py::arg("dim0"), py::arg("dim1"));

    m.def("swapdims", &synapx::swapdims, py::arg("tensor"), py::arg("dim0"), py::arg("dim1"));
    
    m.def("movedim", &synapx::movedim, py::arg("tensor"), py::arg("source"), py::arg("destination"));
    
    m.def("concat", &synapx::concat, py::arg("tensor"), py::arg("dim") = 0);
    
    m.def("stack", &synapx::stack, py::arg("tensor"), py::arg("dim") = 0);
    
    m.def("unbind", &synapx::unbind, py::arg("tensor"), py::arg("dim") = 0);
}
