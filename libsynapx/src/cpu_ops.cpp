

#include <torch/torch.h>


namespace cpu
{

torch::Tensor add_forward(const torch::Tensor& t1, const torch::Tensor& t2) {
    return torch::add(t1, t2);
}

torch::Tensor add_backward(const torch::Tensor& grad, const torch::Tensor& t1_shape, const torch::Tensor& t2_shape) {
    
} 


} // namespace cpu
