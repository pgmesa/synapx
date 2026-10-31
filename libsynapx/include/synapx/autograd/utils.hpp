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

    // Sliding windows of (N, C, H, W) as a view of the padded input, with shape
    // (lH, lW, N, C, kH, kW). From synapgrad's conv_tools.extract_windows.
    torch::Tensor extract_windows(const torch::Tensor& a, torch::IntArrayRef kernel_size, torch::IntArrayRef stride,
                                  torch::IntArrayRef padding, torch::IntArrayRef dilation, double pad_value = 0.0);

    // Adds windows of shape (lH, lW, N, C, kH, kW) back into a (N, C, H, W) tensor,
    // one window at a time. From synapgrad's conv_tools.place_windows.
    torch::Tensor place_windows(const torch::Tensor& windows, torch::IntArrayRef out_shape, torch::IntArrayRef kernel_size,
                                torch::IntArrayRef stride, torch::IntArrayRef padding, torch::IntArrayRef dilation);

}

#endif