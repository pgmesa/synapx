
#include <torch/torch.h>
#include <iostream>

#include <synapx/tensor.hpp>


namespace synapx {

// Constructor implementation
Tensor::Tensor(const torch::Tensor& tensor) : data(tensor) {}

size_t Tensor::numel() const {
    return data.numel();
}

size_t Tensor::ndim() const {
    return data.dim();
}

Tensor Tensor::matmul(const Tensor& other) const {
    return Tensor(torch::matmul(this->data, other.data));
}

} // namespace synapx
