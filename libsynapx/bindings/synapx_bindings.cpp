
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
            } else if (py::isinstance<torch::Tensor>(grad)) {
                torch::Tensor grad_tensor = py::cast<torch::Tensor>(grad);
                self.backward(grad_tensor);
            } else {
                throw std::runtime_error("grad must be a synapx::Tensor or torch::Tensor");
            }
        }, py::arg("grad") = py::none(), 
           "Union[None, synapx.Tensor, torch.Tensor]: Computes the gradient of current tensor w.r.t. graph leaves.")
        .def("data", &synapx::Tensor::data)
        .def("torch", &synapx::Tensor::data)
        .def("numpy", [](const synapx::Tensor& self) {
            return torch_to_numpy(self.data());
        });
    
    m.def("tensor", [](py::object data, bool requires_grad, std::string device) {
        synapx::Device dev = string_to_device(device);
        torch::Tensor tensor = pydata_to_torch(data);
        return synapx::Tensor(tensor, requires_grad, dev);
    }, py::arg("data"), py::arg("requires_grad") = false, py::arg("device") = "cpu");
    
    m.def("ones", [](py::iterable shape, bool requires_grad, std::string device) {
        synapx::Device dev = string_to_device(device);
        std::vector<int64_t> dims;
        for (auto item : shape) {
            // Each item should be convertible to int64_t
            dims.push_back(py::cast<int64_t>(item));
        }
        return synapx::Tensor(torch::ones(dims), requires_grad, dev);
    }, py::arg("shape"), py::arg("requires_grad") = false, py::arg("device") = "cpu");

    m.def("zeros", [](py::iterable shape, bool requires_grad, std::string device) {
        synapx::Device dev = string_to_device(device);
        std::vector<int64_t> dims;
        for (auto item : shape) {
            dims.push_back(py::cast<int64_t>(item));
        }
        return synapx::Tensor(torch::zeros(dims), requires_grad, dev);
    }, py::arg("shape"), py::arg("requires_grad") = false, py::arg("device") = "cpu");
        
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
