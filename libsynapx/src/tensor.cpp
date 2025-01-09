
#include <iostream>
#include <vector>
#include <cstddef>

#include ""

#include <torch/torch.h>
#include <synapx/tensor.hpp>


namespace synapx {

// Constructor implementation
Tensor::Tensor(const torch::Tensor& tensor, bool requires_grad=false, Device device=Device::CPU) : data(tensor), _requires_grad(requires_grad), device(device)  {}

size_t Tensor::numel() const {
    return data.numel();
}

size_t Tensor::dim() const {
    return data.dim();
}

std::vector<int64_t> Tensor::shape() const {
    auto sizes = this->data.sizes();
    return std::vector<int64_t>(sizes.begin(), sizes.end());
}

Tensor Tensor::operator+(const Tensor& other) {
    return this->add(other);
}

Tensor Tensor::operator*(const Tensor& other) {
    return this->mul(other);
}

Tensor Tensor::add(const Tensor& other) const {
    return 
}

Tensor Tensor::mul(const Tensor& other) const {
    return Tensor(torch::mul(this->data, other.data));
}

Tensor Tensor::matmul(const Tensor& other) const {
    return Tensor(torch::matmul(this->data, other.data));
}

} // namespace synapx
