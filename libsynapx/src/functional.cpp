
#include "cpu_ops.hpp"

#include <torch/torch.h>
#include <synapx/tensor.hpp>
#include <synapx/functional.hpp>
#include <synapx/device.hpp>

namespace synapx
{

namespace F
{

TensorPtr add(const TensorPtr& t1, const TensorPtr& t2) {
    // TODO: Check input tensors
    // ...

    torch::Tensor out_data;
    if (t1->device().type() == Device::Type::CPU) {
        out_data = cpu::add_forward(t1->data(), t2->data());
    } else {
        Device not_supported_device = t1->device();
        throw std::runtime_error(not_supported_device.to_string() + " device not supported");
    }

    bool req_grad = t1->requires_grad() || t2->requires_grad();
    TensorPtr out = std::make_shared<Tensor>(std::move(out_data), req_grad, t1->device(), "Add");

    if (req_grad) {
        std::function<void()> backward = [t1, t2, out]() mutable {
            std::cout << "[DEBUG] Inside Backward" << std::endl;

            out->set_grad(std::move(torch::ones(out->shape()))); // Only for debugging
            const std::optional<const torch::Tensor> out_grad_ = out->grad();

            std::cout << "[DEBUG] After getting grad" << '\n';

            if (!out_grad_.has_value()) {
                throw std::runtime_error("Attempted to call backward on a Tensor with no gradient");
            }

            std::cout << "[DEBUG] Before gradient" << std::endl;
            
            const torch::Tensor out_grad = out_grad_.value();

            std::cout << "[DEBUG] After gradient" << std::endl;

            torch::Tensor grad_t1, grad_t2;
            if (t1->device().type() == Device::Type::CPU) {
                auto [grad_t1, grad_t2] = cpu::add_backward(out_grad, t1->shape(), t2->shape());
            } else {
                Device not_supported_device = t1->device();
                throw std::runtime_error(not_supported_device.to_string() + " device not supported for backward");
            }

            std::cout << "[DEBUG] After backward" << std::endl;

            if (t1->requires_grad()) {
                if (t1->grad().has_value()) {
                    torch::Tensor updated_grad = t1->grad().value() + grad_t1;
                    t1->set_grad(std::move(updated_grad));
                } else {
                    t1->set_grad(std::move(grad_t1));
                }
            }
            if (t2->requires_grad()) {
                if (t2->grad().has_value()) {
                    torch::Tensor updated_grad = t2->grad().value() + grad_t2;
                    t2->set_grad(std::move(updated_grad));
                } else {
                    t2->set_grad(std::move(grad_t2));
                }
            }

            std::cout << "[DEBUG] After grad updates" << std::endl;
            
        };

        assert(out->operation().has_value() && "Operation must have a value here");
        out->set_grad_fn(BackwardFunction(backward, out->operation().value()));
    }

    return out;
}

} // namespace F

} // namespace synapx
