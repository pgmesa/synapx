
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

} // namespace synapx
