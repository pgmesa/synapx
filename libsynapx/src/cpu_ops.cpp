
#include "cpu_ops.hpp"

#include <torch/torch.h>


namespace cpu
{

torch::Tensor unbroadcast(const torch::Tensor& grad, const std::vector<int64_t>& target_shape) {
    std::cout << "[DEBUG] Inside unbroadcast" << std::endl;
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
    std::cout << "[DEBUG] Returning from unbroadcast" << std::endl;
    return result;
}

torch::Tensor add_forward(const torch::Tensor& t1, const torch::Tensor& t2) {
    return torch::add(t1, t2);
}

std::pair<torch::Tensor, torch::Tensor> add_backward(
    const torch::Tensor& grad,
    const std::vector<int64_t>& t1_shape,
    const std::vector<int64_t>& t2_shape
) {
    std::cout << "[DEBUG] Inside add backward" << std::endl;
    // For addition, gradient is passed straight through but needs to match original tensor shapes
    torch::Tensor grad_t1 = grad; //unbroadcast(grad, t1_shape);
    std::cout << "[DEBUG] Inside add backward 2" << std::endl;
    torch::Tensor grad_t2 = grad; //unbroadcast(grad, t2_shape);
    std::cout << "[DEBUG] Inside add backward 3" << std::endl;
    
    return {grad_t1, grad_t2};
}

} // namespace cpu
