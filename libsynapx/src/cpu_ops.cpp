
#include "cpu_ops.hpp"

#include <torch/torch.h>


namespace cpu
{

torch::Tensor unbroadcast(const torch::Tensor& grad, const std::vector<int64_t>& target_shape) {
    auto grad_shape = grad.sizes();

    // Reduce extra dimensions
    torch::Tensor result = grad;
    while (result.dim() > static_cast<int64_t>(target_shape.size())) {
        result = result.sum(0); // Sum along the first dimension
    }

    // Sum mismatched dimensions
    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (result.size(i) != target_shape[i]) {
            result = result.sum(i, true); // Sum along axis i, keep reduced dimension
        }
    }

    return result;
}

torch::Tensor add_forward(const torch::Tensor& t1, const torch::Tensor& t2) {
    return torch::add(t1, t2);
}

void add_backward(const torch::Tensor& grad, const std::vector<int64_t>& t1_shape, const std::vector<int64_t>& t2_shape,
                            torch::Tensor& grad_t1, torch::Tensor& grad_t2) {
    grad_t1 = unbroadcast(grad.clone(), t1_shape);
    grad_t2 = unbroadcast(grad.clone(), t2_shape);
}

} // namespace cpu
