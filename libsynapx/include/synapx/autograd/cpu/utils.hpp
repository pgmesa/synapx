#ifndef CPU_UTILS_HPP
#define CPU_UTILS_HPP

#include <torch/torch.h>


namespace synapx::autograd::cpu {

    torch::Tensor unbroadcast(torch::Tensor grad, const torch::IntArrayRef& original_shape);
    torch::Tensor expand_dims(torch::Tensor tensor, const torch::IntArrayRef& dim, bool normalized = false);
    std::vector<int64_t> normalize_dims(int64_t tensor_dim, const torch::IntArrayRef& dim);
    torch::Tensor unravel_index(const torch::Tensor& indices, at::IntArrayRef shape);

}

#endif