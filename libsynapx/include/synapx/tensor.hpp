#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <memory>
#include <vector>
#include <cstddef>
#include <tuple>

#include <torch/torch.h>

#include <synapx/core.hpp>
#include <synapx/device.hpp>


namespace synapx {

    namespace autograd { class Function; }
    
    class SYNAPX_API Tensor {
    public:
        // Constructor
        Tensor();
        Tensor(const torch::Tensor& data, bool requires_grad=false, Device device=Device::CPU());

        const torch::Tensor& data() const;
        bool defined() const;
        bool requires_grad() const;
        void requires_grad_(bool _requires_grad);
        const Device& device() const;

        size_t numel() const;
        size_t dim() const;
        std::vector<int64_t> shape() const;

        bool is_leaf() const;
        void retain_grad();
        bool retains_grad() const;
        bool is_floating_point() const;

        torch::Scalar item() const;
        Tensor to(Device device) const;
        Tensor cpu() const;
        Tensor detach() const;

        const torch::Tensor grad() const;
        void set_grad(const torch::Tensor& grad);

        const std::shared_ptr<autograd::Function> grad_fn() const;
        void set_grad_fn(const std::shared_ptr<autograd::Function> grad_fn);

        void backward(const torch::Tensor& grad={});

        Tensor operator+(const Tensor& other) const;
        Tensor operator+(double other) const;
        Tensor operator*(const Tensor& other) const;
        Tensor operator*(double other) const;
        Tensor operator-(const Tensor& other) const;
        Tensor operator-(double other) const;
        Tensor operator/(const Tensor& other) const;
        Tensor operator/(double other) const;
        Tensor operator-() const;

        Tensor& operator+=(const Tensor& other);
        Tensor& operator+=(double other);
        Tensor& operator-=(const Tensor& other);
        Tensor& operator-=(double other);
        Tensor& operator*=(const Tensor& other);
        Tensor& operator*=(double other);
        Tensor& operator/=(const Tensor& other);
        Tensor& operator/=(double other);

        // Subscript operators for indexing
        // Tensor operator[](int index) const;
        // Tensor operator[](const std::vector<int>& indices) const;

        Tensor add(const Tensor& other) const;
        Tensor add(double other) const;
        Tensor sub(const Tensor& other) const;
        Tensor sub(double other) const;
        Tensor mul(const Tensor& other) const;
        Tensor mul(double other) const;
        Tensor pow(const Tensor& exponent) const;
        Tensor pow(double exponent) const;
        Tensor div(const Tensor& other) const;
        Tensor div(double other) const;
        Tensor matmul(const Tensor& other) const;
        Tensor neg() const;

        Tensor& add_(const Tensor& other);
        Tensor& add_(double other);
        Tensor& sub_(const Tensor& other);
        Tensor& sub_(double other);
        Tensor& mul_(const Tensor& other);
        Tensor& mul_(double other);
        Tensor& pow_(const Tensor& exponent);
        Tensor& pow_(double exponent);
        Tensor& div_(const Tensor& other);
        Tensor& div_(double other);
        Tensor& neg_();
        Tensor& zero_();

        Tensor rsub(const Tensor& other) const;
        Tensor rsub(double other) const;
        Tensor rpow(const Tensor& exponent) const;
        Tensor rpow(double exponent) const;
        Tensor rdiv(const Tensor& exponent) const;
        Tensor rdiv(double exponent) const;
        Tensor rmatmul(const Tensor& exponent) const;
        
        Tensor clone() const;
        Tensor exp() const;
        Tensor log() const;
        Tensor sqrt() const;
        Tensor sum(const torch::IntArrayRef& dim = {}, bool keepdim = false) const;
        Tensor mean(const torch::IntArrayRef& dim = {}, bool keepdim = false) const;
        Tensor max() const;
        std::tuple<Tensor, Tensor> max(int64_t dim, bool keepdim = false) const;
        Tensor min() const;
        std::tuple<Tensor, Tensor> min(int64_t dim, bool keepdim = false) const;

        std::string to_string() const;
        static std::string to_string(torch::Tensor tensor);

    private:
        struct Impl;
        std::shared_ptr<Impl> impl_;
    };

} // namespace synapx

#endif // TENSOR_HPP
