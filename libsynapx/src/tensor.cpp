
#include <synapx/tensor.hpp>

#include <iostream>
#include <vector>
#include <cstddef>

#include <torch/torch.h>
#include <spdlog/spdlog.h>

#include <synapx/functional.hpp>
#include <synapx/autograd/engine.hpp>


namespace synapx {
    struct Tensor::Impl {
        Impl() {};
        Impl(const torch::Tensor& data, bool req_grad, Device device)
            : data(data), requires_grad(req_grad), device(device) {};

        torch::Tensor data;
        bool requires_grad;
        Device device;
        bool retains_grad;
        
        torch::Tensor grad;
        std::shared_ptr<autograd::Function> grad_fn;
    };

    Tensor::Tensor() {}

    Tensor::Tensor(const torch::Tensor& data,  bool requires_grad, Device device)
        : impl_(std::make_shared<Impl>(data, requires_grad, device)) {}

    const torch::Tensor& Tensor::data() const { 
        return impl_->data; 
    }

    bool Tensor::requires_grad() const { 
        return impl_->requires_grad; 
    }

    bool Tensor::defined() const {
        return impl_->data.defined();
    }

    const Device& Tensor::device() const { 
        return impl_->device; 
    }

    bool Tensor::is_leaf() const {
        return !requires_grad() || !grad_fn();
    }

    void Tensor::retain_grad() {
        impl_->retains_grad = true;
    }

    bool Tensor::retains_grad() const {
        return impl_->retains_grad;
    }

    const torch::Tensor Tensor::grad() const {
        const torch::Tensor& maybe_grad = impl_->grad;
        if (!this->is_leaf() && !this->retains_grad() && !maybe_grad.defined()) {
            std::cout << (
                "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad "
                "attribute won't be populated during autograd.backward(). If you indeed want the .grad "
                "field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. "
                "If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor "
                "instead."
            ) << std::endl;
        } 
        return maybe_grad;
    }

    const std::shared_ptr<autograd::Function> Tensor::grad_fn() const {
        return impl_->grad_fn; 
    }

    void Tensor::set_grad(const torch::Tensor& grad) {
        impl_->grad = grad;
    }

    void Tensor::set_grad_fn(const std::shared_ptr<autograd::Function> grad_fn) {
        impl_->grad_fn = grad_fn;
    }

    size_t Tensor::numel() const {
        return impl_->data.numel();
    }

    size_t Tensor::dim() const {
        return impl_->data.dim();
    }

    std::vector<int64_t> Tensor::shape() const {
        auto sizes = impl_->data.sizes();
        return std::vector<int64_t>(sizes.begin(), sizes.end());
    }

    void Tensor::backward(const torch::Tensor& grad) {
        if (impl_->grad_fn) {
            spdlog::debug("Backward called");
            torch::Tensor grad_ = grad;
            if (!grad_.defined()) grad_ = torch::ones_like(data());
            autograd::backward(impl_->grad_fn, grad_);
        } else {
            spdlog::warn("No backward function defined for this tensor");
        }
    }

    Tensor Tensor::operator+(const Tensor& other) const {
        return add(other);
    }

    Tensor Tensor::operator*(const Tensor& other) const {
        return mul(other);
    }

    Tensor Tensor::operator-(const Tensor& other) const {
        return add((-other));
    }

    Tensor Tensor::operator/(const Tensor& other) const {
        return mul(other.pow(-1.0));
    }

    Tensor Tensor::operator-() const {
        return mul(Tensor(torch::tensor(-1.0), false, device()));
    }

    Tensor Tensor::add(const Tensor& other) const {
        return F::add(*this, other);
    }

    Tensor Tensor::mul(const Tensor& other) const {
        return F::mul(*this, other);
    }

    Tensor Tensor::matmul(const Tensor& other) const {
        return F::matmul(*this, other);
    }

    Tensor Tensor::pow(const Tensor& exponent) const {
        return F::pow(*this, exponent);
    }

    Tensor Tensor::pow(double exponent) const {
        return F::pow(*this, exponent);
    }

    std::string Tensor::to_string() const {
        std::stringstream ss;
        ss << impl_->data;
        return ss.str();
    }

    std::string Tensor::to_string(torch::Tensor tensor) {
        std::stringstream ss;
        ss << tensor;
        return ss.str();
    }
    
} // namespace synapx
