
#include <synapx/functional.hpp>

#include <vector>

#include <torch/torch.h>

#include <synapx/tensor.hpp>
#include <synapx/device.hpp>
#include <synapx/autograd/cpu/ops.hpp>


namespace synapx::F {

    Tensor add(const Tensor& t1, const Tensor& t2) {
        std::vector<Tensor> outs = detail::dispatch_op(
            {t1, t2}, 
            [](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Add>();
                throw std::runtime_error("Add: unsupported device");
            }
        );
        return outs[0];
    }

    Tensor add(const Tensor& t1, double t2) {
        return add(t1, Tensor(torch::tensor(t2, t1.data().options()), false, t1.device()));
    }

    Tensor sub(const Tensor& t1, const Tensor& t2) {
        return add(t1, t2 * -1);
    }

    Tensor sub(const Tensor& t1, double t2) {
        return sub(t1, Tensor(torch::tensor(t2, t1.data().options()), false, t1.device()));
    }

    Tensor mul(const Tensor& t1, const Tensor& t2) {
        std::vector<Tensor> outs = detail::dispatch_op(
            {t1, t2}, 
            [](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Mul>();
                throw std::runtime_error("Mul: unsupported device");
            }
        );
        return outs[0];
    }

    Tensor mul(const Tensor& t1, double t2) {
        return mul(t1, Tensor(torch::tensor(t2, t1.data().options()), false, t1.device()));
    }

    Tensor div(const Tensor& t1, const Tensor& t2) {
        std::vector<Tensor> outs = detail::dispatch_op(
            {t1, t2}, 
            [](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Div>();
                throw std::runtime_error("Div: unsupported device");
            }
        );
        return outs[0];
    }

    Tensor div(const Tensor& t1, double t2) {
        return div(t1, Tensor(torch::tensor(t2, t1.data().options()), false, t1.device()));
    }

    Tensor matmul(const Tensor& t1, const Tensor& t2) {
        std::vector<Tensor> outs = detail::dispatch_op(
            {t1, t2}, 
            [](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Matmul>();
                throw std::runtime_error("Matmul: unsupported device");
            }
        );
        return outs[0];
    }

    Tensor pow(const Tensor& t1, const Tensor& exp) {
        std::vector<Tensor> outs = detail::dispatch_op(
            {t1, exp}, 
            [](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Pow>();
                throw std::runtime_error("Pow: unsupported device");
            }
        );
        return outs[0];
    }

    Tensor pow(const Tensor& t1, double exp) {
        return pow(t1, Tensor(torch::tensor(exp, t1.data().options()), false, t1.device()));
    }

    Tensor neg(const Tensor& t1) {
        return mul(t1, Tensor(torch::tensor(-1.0, t1.data().options()), false, t1.device()));
    }


    // Reverse Functions
    Tensor rsub(const Tensor& t1, const Tensor& t2) {
        return sub(t2, t1);
    };

    Tensor rsub(const Tensor& t1, double t2) {
        return sub(Tensor(torch::tensor(t2, t1.data().options()), false, t1.device()), t1);
    };

    Tensor rpow(const Tensor& exp, const Tensor& base) {
        return pow(base, exp);
    };

    Tensor rpow(const Tensor& exp, double base) {
        return rpow(exp, Tensor(torch::tensor(base, exp.data().options()), false, exp.device()));
    };

    Tensor rdiv(const Tensor& t1, const Tensor& t2) {
        return div(t2, t1);
    };

    Tensor rdiv(const Tensor& t1, double t2) {
        return div(Tensor(torch::tensor(t2, t1.data().options()), false, t1.device()), t1);
    };

    Tensor rmatmul(const Tensor& t1, const Tensor& t2) {
        return matmul(t2, t1);
    };


    // Other functions
    Tensor addmm(const Tensor& inp, const Tensor& mat1, const Tensor& mat2) {
        std::vector<Tensor> outs = detail::dispatch_op(
            {inp, mat1, mat2}, 
            [](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Addmm>();
                throw std::runtime_error("Addmm: unsupported device");
            }
        );
        return outs[0];
    }

    Tensor clone(const Tensor& t) {
        std::vector<Tensor> outs = detail::dispatch_op(
            {t}, 
            [](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Clone>();
                throw std::runtime_error("Clone: unsupported device");
            }
        );
        return outs[0];
    }

    Tensor exp(const Tensor& t) {
        std::vector<Tensor> outs = detail::dispatch_op(
            {t}, 
            [](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Exp>();
                throw std::runtime_error("Exp: unsupported device");
            }
        );
        return outs[0];
    }

    Tensor log(const Tensor& t) {
        std::vector<Tensor> outs = detail::dispatch_op(
            {t}, 
            [](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Log>();
                throw std::runtime_error("Log: unsupported device");
            }
        );
        return outs[0];
    }

    Tensor sqrt(const Tensor& t) {
        std::vector<Tensor> outs = detail::dispatch_op(
            {t}, 
            [](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Sqrt>();
                throw std::runtime_error("Sqrt: unsupported device");
            }
        );
        return outs[0];
    }

    Tensor sum(const Tensor& t, const torch::IntArrayRef& dim, bool keepdim) {
        std::vector<Tensor> outs = detail::dispatch_op(
            {t}, 
            [dim, keepdim](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Sum>(dim, keepdim);
                throw std::runtime_error("Sum: unsupported device");
            }
        );
        return outs[0];
    }

    Tensor mean(const Tensor& t, const torch::IntArrayRef& dim, bool keepdim) {
        std::vector<Tensor> outs = detail::dispatch_op(
            {t}, 
            [dim, keepdim](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Mean>(dim, keepdim);
                throw std::runtime_error("Mean: unsupported device");
            }
        );
        return outs[0];
    }

} // namespace synapx
