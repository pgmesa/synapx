
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <torch/extension.h>
#include <torch/torch.h>
#include <synapx/tensor.hpp>

#include "parsers.hpp"


namespace py = pybind11;


PYBIND11_MODULE(_C, m) {
    m.doc() = "Synapx core C++ bindings";

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

    py::class_<synapx::Tensor>(m, "Tensor")
        .def("numel", &synapx::Tensor::numel)
        .def("dim", &synapx::Tensor::dim)
        .def_property_readonly("shape", &synapx::Tensor::shape)
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
        .def("add", [](const synapx::Tensor& self, py::object other) {
            synapx::Tensor other_tensor = pyobject_to_synapx(other, self.device());
            return self + other_tensor;
        }, py::arg("other"))
        .def("mul", [](const synapx::Tensor& self, py::object other) {
            synapx::Tensor other_tensor = pyobject_to_synapx(other, self.device());
            return self * other_tensor;
        }, py::arg("other"))
        .def("matmul", [](const synapx::Tensor& self, py::object other) {
            synapx::Tensor other_tensor = pyobject_to_synapx(other, self.device());
            return self.matmul(other_tensor);
        }, py::arg("other"))
        .def("pow", [](const synapx::Tensor& self, py::object exponent) {
            if (py::isinstance<py::float_>(exponent) || py::isinstance<py::int_>(exponent)) {
                double exp_val = py::cast<double>(exponent);
                return self.pow(exp_val);
            } else {
                synapx::Tensor exp_tensor = pyobject_to_synapx(exponent, self.device());
                return self.pow(exp_tensor);
            }
        }, py::arg("exponent"))
        .def("requires_grad", &synapx::Tensor::requires_grad)
        .def("is_leaf", &synapx::Tensor::is_leaf)
        .def("retain_grad", &synapx::Tensor::retain_grad)
        .def("retains_grad", &synapx::Tensor::retains_grad)
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
        return t1.add(t2);
    }, py::arg("t1"), py::arg("t2"));
    
    m.def("mul", [](synapx::Tensor t1, synapx::Tensor t2) {
        return t1.mul(t2);
    }, py::arg("t1"), py::arg("t2"));
    
    m.def("matmul", [](synapx::Tensor t1, synapx::Tensor t2) {
        return t1.matmul(t2);
    }, py::arg("t1"), py::arg("t2"));
}
