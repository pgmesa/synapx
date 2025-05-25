#ifndef UTILS_HPP
#define UTILS_HPP

#include <torch/torch.h>


namespace synapx::autograd::utils {

    torch::Tensor unbroadcast(const torch::Tensor& grad, const std::vector<int64_t>& original_shape);
    torch::Tensor unbroadcast(const torch::Tensor& grad, torch::IntArrayRef& original_shape);

}

#endif