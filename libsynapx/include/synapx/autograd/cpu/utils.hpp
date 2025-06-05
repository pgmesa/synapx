#ifndef CPU_UTILS_HPP
#define CPU_UTILS_HPP

#include <torch/torch.h>


namespace synapx::autograd::cpu {

    torch::Tensor unbroadcast(torch::Tensor grad, const torch::IntArrayRef& original_shape);
    torch::Tensor expand_dims(torch::Tensor tensor, const torch::IntArrayRef& dims);
}

#endif