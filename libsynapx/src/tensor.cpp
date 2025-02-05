
#include <iostream>
#include <vector>
#include <cstddef>

#include <torch/torch.h>
#include <synapx/tensor.hpp>
#include <synapx/functional.hpp>


namespace synapx {

// Constructor implementation
Tensor::Tensor(const torch::Tensor& tensor, bool requires_grad, Device device, std::optional<std::string> operation)
    : _data(std::move(tensor)), _requires_grad(requires_grad), _device(device), _operation(operation), _grad(std::nullopt), _grad_fn(std::nullopt) {}


const torch::Tensor& Tensor::data() const { return this->_data; }
const bool Tensor::requires_grad() const { return this->_requires_grad; }
const Device Tensor::device() const { return this->_device; }
const std::optional<const std::string> Tensor::operation() const { return this->_operation; }
const std::optional<const torch::Tensor>& Tensor::grad() const { return this->_grad; }
const std::optional<const BackwardFunction>& Tensor::grad_fn() const { return this->_grad_fn; }

void Tensor::set_grad(const torch::Tensor& grad) const {
    this->_grad = grad;
}

void Tensor::set_grad_fn(const BackwardFunction& grad_fn) {
    this->_grad_fn = std::move(grad_fn);
}


size_t Tensor::numel() const {
    return this->_data.numel();
}

size_t Tensor::dim() const {
    return this->_data.dim();
}

std::vector<int64_t> Tensor::shape() const {
    auto sizes = this->_data.sizes();
    return std::vector<int64_t>(sizes.begin(), sizes.end());
}

void Tensor::backward(const std::optional<const Tensor>& grad) {
    if (this->_grad_fn.has_value()) {
        std::cout << "[DEBUG] Backward called" << std::endl;
        if (grad.has_value()) {
            set_grad(grad.value().data());
            std::cout << "[DEBUG] Gradient Set" << std::endl;
        }
        std::function<void()> backward_fn = this->_grad_fn.value();
        backward_fn();
    } else {
        std::cerr << "Warning: No backward function defined for this tensor." << std::endl;
    }
}

TensorPtr Tensor::operator+(const TensorPtr& other) {
    return this->add(other);
}

// Tensor Tensor::operator*(const Tensor& other) {
//     return this->mul(other);
// }

TensorPtr Tensor::add(const TensorPtr& other) {
    return F::add(shared_from_this(), other);
}

// Tensor Tensor::mul(const Tensor& other) const {
//     return F::mul(this->data, other);
// }

// Tensor Tensor::matmul(const Tensor& other) const {
//     return F::matmul(this->data, other);
//}

std::string Tensor::to_string() const {
    return this->_data.toString();
}

// --------------------------------------------------------------------------------

BackwardFunction::BackwardFunction(std::function<void()> backward, std::string operation)
    : backward(std::move(backward)), operation(operation) {}


void BackwardFunction::operator()() const {
    if (this->backward) {
        this->backward();
    } else {
        throw std::runtime_error("Attempted to call an empty BackwardFunction");
    }
}

std::string BackwardFunction::name() const {
    return this->operation + "Backward";
}

std::string BackwardFunction::to_string() const {
    return "<" + this->name() + ">";
}

// --------------------------------------------------------------------------------

} // namespace synapx
