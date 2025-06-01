
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
        return pow(t1, Tensor(torch::tensor(exp), false, t1.device()));
    }

} // namespace synapx
