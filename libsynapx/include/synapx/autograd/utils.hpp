#ifndef CPU_UTILS_HPP
#define CPU_UTILS_HPP

#include <torch/torch.h>
#include <synapx/tensor.hpp>


namespace synapx::autograd {

    synapx::Tensor unbroadcast(synapx::Tensor grad, torch::IntArrayRef original_shape);
    synapx::Tensor expand_dims(synapx::Tensor tensor, torch::IntArrayRef dim, bool normalized = false);
    std::vector<int64_t> normalize_dims(int64_t tensor_dim, torch::IntArrayRef dim);

    /**
     * @brief 
     * Converts a tensor of flat indices into a tensor of coordinate vectors. This is a 
     * `libtorch` implementation of `numpy.unravel_index`.
     * 
     * Source: https://github.com/pytorch/pytorch/issues/35674#issuecomment-1741608630
     * 
     * @param indices
     * @param shape 
     * @return torch::Tensor 
     */
    torch::Tensor unravel_index(const torch::Tensor& indices, at::IntArrayRef shape);

}

#endif