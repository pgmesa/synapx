
#ifndef CPU_OPS_HPP
#define CPU_OPS_HPP

#include <torch/torch.h>


namespace cpu
{

torch::Tensor add_forward(const torch::Tensor& t1, const torch::Tensor& t2);

std::pair<torch::Tensor, torch::Tensor> add_backward(
    const torch::Tensor& grad, 
    const std::vector<int64_t>& t1_shape, 
    const std::vector<int64_t>& t2_shape
);

} // namespace cpu

#endif 