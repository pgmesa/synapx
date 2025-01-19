
#ifndef CPU_OPS_HPP
#define CPU_OPS_HPP

#include <torch/torch.h>


namespace cpu
{

torch::Tensor add_forward(const torch::Tensor& t1, const torch::Tensor& t2);

void add_backward(const torch::Tensor& grad, const std::vector<int64_t>& t1_shape, const std::vector<int64_t>& t2_shape,
                    torch::Tensor& grad_t1, torch::Tensor& grad_t2);

} // namespace cpu

#endif 