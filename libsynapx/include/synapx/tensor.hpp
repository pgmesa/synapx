#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <memory>
#include <vector>
#include <cstddef>  // for size_t

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

        Tensor detach() const;
        torch::Scalar item() const;
        void zero_();

        size_t numel() const;
        size_t dim() const;
        std::vector<int64_t> shape() const;

        bool is_leaf() const;
        void retain_grad();
        bool retains_grad() const;
        bool is_floating_point() const;

        Tensor to(Device device) const;
        Tensor cpu() const;

        const torch::Tensor grad() const;
        void set_grad(const torch::Tensor& grad);

        const std::shared_ptr<autograd::Function> grad_fn() const;
        void set_grad_fn(const std::shared_ptr<autograd::Function> grad_fn);

        void backward(const torch::Tensor& grad={});

        Tensor operator+(const Tensor& other) const;
        Tensor operator*(const Tensor& other) const;
        Tensor operator-(const Tensor& other) const;
        Tensor operator/(const Tensor& other) const;
        Tensor operator-() const;

        Tensor add(const Tensor& other) const;
        Tensor mul(const Tensor& other) const;
        Tensor matmul(const Tensor& other) const;
        Tensor pow(const Tensor& exponent) const;
        Tensor pow(double exponent) const;
        Tensor clone() const;
        Tensor exp() const;
        Tensor log() const;
        Tensor sqrt() const;
        Tensor sum(const torch::IntArrayRef& dim = {}, bool keepdim = false) const;

        std::string to_string() const;
        static std::string to_string(torch::Tensor tensor);

    private:
        struct Impl;
        std::shared_ptr<Impl> impl_;
    };

} // namespace synapx

#endif // TENSOR_HPP
