
#include <synapx/tensor.hpp>

#include <memory>
#include <vector>
#include <cstddef>

#include <torch/torch.h>
#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include <synapx/functional.hpp>
#include <synapx/autograd/engine.hpp>


namespace synapx {

    namespace {
        void in_place_check(const Tensor& tensor) {
            if (autograd::is_grad_enabled() && tensor.requires_grad() && tensor.is_leaf())
                throw std::runtime_error(
                    "A leaf Tensor that requires grad is being " 
                    "used in an in-place operation"
                );
        }
    }
    struct Tensor::Impl {
        Impl() {};
        Impl(const torch::Tensor& data, bool req_grad)
            : data(data.detach()), requires_grad(req_grad) {};

        torch::Tensor data;
        bool requires_grad;

        bool retains_grad;
        uint32_t output_nr = 0;
        
        torch::Tensor grad;
        std::shared_ptr<autograd::Node> grad_fn;
    };

    Tensor::Tensor() : impl_(std::make_shared<Impl>()) {}

    Tensor::Tensor(const torch::Tensor& data,  bool requires_grad)
        : impl_(std::make_shared<Impl>(data, requires_grad && autograd::is_grad_enabled())) {}


    void Tensor::set_output_nr(uint32_t nr) {
        impl_->output_nr = nr;
    }

    uint32_t Tensor::output_nr() const {
        return impl_->output_nr;
    }

    torch::Tensor Tensor::data() const { 
        return impl_->data.detach();
    }

    void Tensor::set_data(const Tensor& other) {
        impl_->data = other.data();
    }

    bool Tensor::requires_grad() const { 
        return impl_->requires_grad; 
    }

    Tensor& Tensor::requires_grad_(bool _requires_grad) {
        if (!is_leaf()) {
            throw std::runtime_error(
                "You can only change requires_grad flags of leaf variables. "
                "If you want to use a computed variable in a subgraph that doesn't require "
                "differentiation use var_no_grad = var.detach()"
            );
        }
            
        if (_requires_grad && !is_floating_point()){
            throw std::runtime_error("Only floating point Tensors can require gradients");
        }
            
        impl_->requires_grad = _requires_grad;
        return *this;
    }

    torch::TensorOptions Tensor::options() const {
        return impl_->data.options();
    }

    bool Tensor::is_floating_point() const {
        return impl_->data.is_floating_point();
    }

    size_t Tensor::numel() const {
        return impl_->data.numel();
    }

    size_t Tensor::dim() const {
        return impl_->data.dim();
    }

    size_t Tensor::size(int64_t dim) const {
        return impl_->data.size(dim);
    }

    torch::IntArrayRef Tensor::sizes() const {
        return impl_->data.sizes();
    }

    IntArray Tensor::shape() const {
        return sizes().vec();
    }

    Tensor Tensor::detach() const {
        return Tensor(impl_->data, false);
    }

    Tensor Tensor::count_nonzero() const {
        return Tensor(impl_->data.count_nonzero(), false);
    }

    Tensor Tensor::argmax(std::optional<int64_t> dim, bool keepdim) const {
        return Tensor(impl_->data.argmax(dim, keepdim));
    };

    Tensor Tensor::argmin(std::optional<int64_t> dim, bool keepdim) const {
        return Tensor(impl_->data.argmin(dim, keepdim));
    };

    torch::Scalar Tensor::item() const {
        return impl_->data.item();
    }

    bool Tensor::defined() const {
        return impl_->data.defined();
    }

    torch::Dtype Tensor::dtype() const { 
        return impl_->data.dtype().toScalarType(); 
    }

    torch::Device Tensor::device() const { 
        return impl_->data.device(); 
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

    const synapx::Tensor Tensor::grad() const {
        const torch::Tensor& maybe_grad = impl_->grad;
        if (!this->is_leaf() && !this->retains_grad() && !maybe_grad.defined()) {
            spdlog::warn(
                "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad "
                "attribute won't be populated during autograd.backward(). If you indeed want the .grad "
                "field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. "
                "If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor "
                "instead."
            );
        }
        if (maybe_grad.defined()) {
            return Tensor(maybe_grad);
        } else {
            return Tensor();
        }
    }

    std::shared_ptr<autograd::Node> Tensor::grad_fn() const {
        return impl_->grad_fn;
    }

    void Tensor::set_grad(const Tensor& grad) {
        impl_->grad = grad.data();
    }

    void Tensor::set_grad_fn(std::shared_ptr<autograd::Node> grad_fn) {
        impl_->grad_fn = grad_fn;
    }

    void Tensor::backward(const Tensor& grad) {
        autograd::run_backward(*this, grad);
    }

    // Operators
    Tensor Tensor::operator+(const Tensor& other) const {
        return add(other);
    }

    Tensor Tensor::operator+(double other) const {
        return add(other);
    }

    Tensor Tensor::operator-(const Tensor& other) const {
        return sub(other);
    }

    Tensor Tensor::operator-(double other) const {
        return sub(other);
    }

    Tensor Tensor::operator*(const Tensor& other) const {
        return mul(other);
    }

    Tensor Tensor::operator*(double other) const {
        return mul(other);
    }

    Tensor Tensor::operator/(const Tensor& other) const {
        return div(other);
    }

    Tensor Tensor::operator/(double other) const {
        return div(other);
    }

    Tensor Tensor::operator-() const {
        return neg();
    }

    Tensor Tensor::operator[](int64_t index) const {
        return slice({index});
    }

    Tensor Tensor::operator[](const TensorIndices& indices) const {
        return slice(indices);
    }
    
    Tensor Tensor::operator==(const Tensor& other) const {
        return Tensor(impl_->data == other.data());
    }

    Tensor Tensor::operator!=(const Tensor& other) const {
        return Tensor(impl_->data != other.data());
    }

    Tensor Tensor::operator<(const Tensor& other) const {
        return Tensor(impl_->data < other.data());
    }

    Tensor Tensor::operator<=(const Tensor& other) const {
        return Tensor(impl_->data <= other.data());
    }

    Tensor Tensor::operator>(const Tensor& other) const {
        return Tensor(impl_->data > other.data());
    }

    Tensor Tensor::operator>=(const Tensor& other) const {
        return Tensor(impl_->data >= other.data());
    }

    Tensor Tensor::operator==(double other) const {
        return Tensor(impl_->data == other);
    }

    Tensor Tensor::operator!=(double other) const {
        return Tensor(impl_->data != other);
    }

    Tensor Tensor::operator<(double other) const {
        return Tensor(impl_->data < other);
    }

    Tensor Tensor::operator<=(double other) const {
        return Tensor(impl_->data <= other);
    }

    Tensor Tensor::operator>(double other) const {
        return Tensor(impl_->data > other);
    }

    Tensor Tensor::operator>=(double other) const {
        return Tensor(impl_->data >= other);
    }

    // Inplace operators
    Tensor& Tensor::operator+=(const Tensor& other) {
        return add_(other);
    }

    Tensor& Tensor::operator+=(double other) {
        return add_(other);
    }

    Tensor& Tensor::operator-=(const Tensor& other) {
        return sub_(other);
    }

    Tensor& Tensor::operator-=(double other) {
        return sub_(other);
    }

    Tensor& Tensor::operator*=(const Tensor& other) {
        return mul_(other);
    }

    Tensor& Tensor::operator*=(double other) {
        return mul_(other);
    }

    Tensor& Tensor::operator/=(const Tensor& other) {
        return div_(other);
    }

    Tensor& Tensor::operator/=(double other) {
        return div_(other);
    }

    // Funtions
    Tensor Tensor::add(const Tensor& other) const {
        return synapx::add(*this, other);
    }

    Tensor Tensor::add(double other) const {
        return synapx::add(*this, other);
    }

    Tensor Tensor::sub(const Tensor& other) const {
        return synapx::sub(*this, other);
    }

    Tensor Tensor::sub(double other) const {
        return synapx::sub(*this, other);
    }

    Tensor Tensor::mul(const Tensor& other) const {
        return synapx::mul(*this, other);
    }

    Tensor Tensor::mul(double other) const {
        return synapx::mul(*this, other);
    }

    Tensor Tensor::div(const Tensor& other) const {
        return synapx::div(*this, other);
    }

    Tensor Tensor::div(double other) const {
        return synapx::div(*this, other);
    }

    Tensor Tensor::matmul(const Tensor& other) const {
        return synapx::matmul(*this, other);
    }

    Tensor Tensor::pow(const Tensor& exponent) const {
        return synapx::pow(*this, exponent);
    }

    Tensor Tensor::pow(double exponent) const {
        return synapx::pow(*this, exponent);
    }

    Tensor Tensor::neg() const {
        return synapx::neg(*this);
    }

    // In-place functions
    Tensor& Tensor::add_(const Tensor& other) {
        in_place_check(*this);
        impl_->data.add_(other.data());
        return *this;
    }

    Tensor& Tensor::add_(double other) {
        in_place_check(*this);
        impl_->data.add_(other);
        return *this;
    }

    Tensor& Tensor::sub_(const Tensor& other) {
        in_place_check(*this);
        impl_->data.sub_(other.data());
        return *this;
    }

    Tensor& Tensor::sub_(double other) {
        in_place_check(*this);
        impl_->data.sub_(other);
        return *this;
    }

    Tensor& Tensor::mul_(const Tensor& other) {
        in_place_check(*this);
        impl_->data.mul_(other.data());
        return *this;
    }

    Tensor& Tensor::mul_(double other) {
        in_place_check(*this);
        impl_->data.mul_(other);
        return *this;
    }

    Tensor& Tensor::pow_(const Tensor& exponent) {
        in_place_check(*this);
        impl_->data.pow_(exponent.data());
        return *this;
    }

    Tensor& Tensor::pow_(double exponent) {
        in_place_check(*this);
        impl_->data.pow_(exponent);
        return *this;
    }

    Tensor& Tensor::div_(const Tensor& other) {
        in_place_check(*this);
        impl_->data.div_(other.data());
        return *this;
    }

    Tensor& Tensor::div_(double other) {
        in_place_check(*this);
        impl_->data.div_(other);
        return *this;
    }

    Tensor& Tensor::neg_() {
        in_place_check(*this);
        impl_->data.neg_();
        return *this;
    }

    Tensor& Tensor::zero_() {
        in_place_check(*this);
        impl_->data.zero_();
        return *this;
    }

    Tensor& Tensor::fill_(double value) {
        in_place_check(*this);
        impl_->data.fill_(value);
        return *this;
    }

    Tensor& Tensor::uniform_(double from, double to) {
        in_place_check(*this);
        impl_->data.uniform_(from, to);
        return *this;
    }

    Tensor& Tensor::normal_(double mean, double std) {
        in_place_check(*this);
        impl_->data.normal_(mean, std);
        return *this;
    }


    Tensor& Tensor::unsqueeze_(int64_t dim) {
        in_place_check(*this);
        impl_->data.unsqueeze_(dim);
        return *this;
    }

    Tensor& Tensor::scatter_(int64_t dim, const Tensor& index, double value) {
        in_place_check(*this);
        impl_->data.scatter_(dim, index.data(), value);
        return *this;
    }

    Tensor& Tensor::index_put_(const TensorIndices& idx, const Tensor& value) {
        in_place_check(*this);
        impl_->data.index_put_(idx, value.data());
        return *this;
    }

    Tensor& Tensor::index_put_(const TensorIndices& idx, double value) {
        in_place_check(*this);
        impl_->data.index_put_(idx, value);
        return *this;
    }

    Tensor& Tensor::copy_(const Tensor& src) {
        in_place_check(*this);
        impl_->data.copy_(src.data());
        return *this;
    }

    Tensor& Tensor::to_(torch::Device device) {
        in_place_check(*this);
        impl_->data = impl_->data.to(device);
        return *this;
    }

    Tensor& Tensor::to_(torch::Dtype dtype) {
        in_place_check(*this);
        impl_->data = impl_->data.to(dtype);
        return *this;
    }

    // Reverse functions
    Tensor Tensor::rsub(const Tensor& other) const {
        return synapx::rsub(*this, other);
    }

    Tensor Tensor::rsub(double other) const {
        return synapx::rsub(*this, other);
    }

    Tensor Tensor::rpow(const Tensor& exponent) const {
        return synapx::rpow(*this, exponent);
    }

    Tensor Tensor::rpow(double exponent) const {
        return synapx::rpow(*this, exponent);
    }
    
    Tensor Tensor::rdiv(const Tensor& other) const {
        return synapx::rdiv(*this, other);
    }

    Tensor Tensor::rdiv(double other) const {
        return synapx::rdiv(*this, other);
    }

    Tensor Tensor::rmatmul(const Tensor& other) const {
        return synapx::matmul(*this, other);
    }

    // Other functions
    Tensor Tensor::to(torch::Device device) const {
        return synapx::copy_to(*this, device);
    }

    Tensor Tensor::to(torch::Dtype dtype) const {
        return synapx::copy_to(*this, dtype);
    }

    Tensor Tensor::cpu() const {
        return to(torch::Device(torch::DeviceType::CPU));
    }

    Tensor Tensor::cuda(int8_t index) const {
        return to(torch::Device(torch::DeviceType::CUDA, index));
    }


    Tensor Tensor::clone() const {
        return synapx::clone(*this);
    }

    Tensor Tensor::exp() const {
        return synapx::exp(*this);
    }

    Tensor Tensor::log() const {
        return synapx::log(*this);
    }

    Tensor Tensor::sqrt() const {
        return synapx::sqrt(*this);
    }

    Tensor Tensor::sum(torch::IntArrayRef dim, bool keepdim) const {
        return synapx::sum(*this, dim, keepdim);
    }

    Tensor Tensor::mean(torch::IntArrayRef dim, bool keepdim) const {
        return synapx::mean(*this, dim, keepdim);
    }

    Tensor Tensor::max() const {
        return synapx::max(*this);
    }

    std::tuple<Tensor, Tensor> Tensor::max(int64_t dim, bool keepdim) const {
        return synapx::max(*this, dim, keepdim);
    }

    Tensor Tensor::min() const {
        return synapx::min(*this);
    }

    std::tuple<Tensor, Tensor> Tensor::min(int64_t dim, bool keepdim) const {
        return synapx::min(*this, dim, keepdim);
    }

    Tensor Tensor::squeeze(torch::IntArrayRef dim) const {
        return synapx::squeeze(*this, dim);
    }
    
    Tensor Tensor::unsqueeze(int64_t dim) const {
        return synapx::unsqueeze(*this, dim);
    }

    Tensor Tensor::reshape(torch::IntArrayRef shape) const {
        return synapx::reshape(*this, shape);
    }

    Tensor Tensor::broadcast_to(torch::IntArrayRef shape) const {
        return synapx::broadcast_to(*this, shape);
    }

    Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
        return synapx::transpose(*this, dim0, dim1);
    }

    Tensor Tensor::t() const {
        if (dim() > 2) {
            std::string msg = fmt::format(
                "t() expects a tensor with <= 2 dimensions, but self is {}D", dim()
            );
            throw std::runtime_error(msg);
        }
        return synapx::transpose(*this, 0, 1);
    }

    Tensor Tensor::swapdims(int64_t dim0, int64_t dim1) const {
        return synapx::swapdims(*this, dim0, dim1);
    }

    Tensor Tensor::movedim(int64_t src, int64_t dest) const {
        return synapx::movedim(*this, src, dest);
    }

    Tensor Tensor::slice(const TensorIndices& indices) const {
        return synapx::slice(*this, indices);
    }

    Tensor Tensor::select(int64_t dim, int64_t index) const {
        return synapx::select(*this, dim, index);
    }

    Tensor Tensor::where(const Tensor& condition, const Tensor& other) const {
       return synapx::where(condition, *this, other); 
    }

    Tensor Tensor::where(const Tensor& condition, double other) const {
        return synapx::where(condition, *this, other);
    }


    Tensor Tensor::relu() const {
        return synapx::relu(*this);
    }

    Tensor Tensor::sigmoid() const {
        return synapx::sigmoid(*this);
    }

    Tensor Tensor::softmax(int64_t dim) const {
        return synapx::softmax(*this, dim);
    }
    
    Tensor Tensor::log_softmax(int64_t dim) const {
        return synapx::log_softmax(*this, dim);
    }

    Tensor Tensor::flatten(int64_t start_dim, int64_t end_dim) const {
        return synapx::flatten(*this, start_dim, end_dim);
    }


    TensorIterator Tensor::begin() const { 
        return TensorIterator(*this, 0); 
    }

    TensorIterator Tensor::end() const { 
        return TensorIterator(*this, size(0)); 
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


    Tensor operator+(double scalar, const Tensor& tensor) {
        return tensor.add(scalar);
    }

    Tensor operator*(double scalar, const Tensor& tensor) {
        return tensor.mul(scalar);
    }

    Tensor operator-(double scalar, const Tensor& tensor) {
        return tensor.rsub(scalar);
    }

    Tensor operator/(double scalar, const Tensor& tensor) {
        return tensor.rdiv(scalar);
    }
    
    
} // namespace synapx
