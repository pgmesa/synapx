#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <memory>
#include <vector>
#include <cstddef>  // for size_t

#include <torch/torch.h>
#include <synapx/core.hpp>
#include <synapx/device.hpp>


namespace synapx {

class Tensor; // Forward declaration
using TensorPtr = std::shared_ptr<Tensor>;

class SYNAPX_API BackwardFunction {

public:
    std::function<void()> backward;
    std::string operation;

    BackwardFunction(std::function<void()> backward, std::string operation);

    void operator()() const;

    std::string name() const;

    std::string to_string() const;

};

class SYNAPX_API Tensor: public std::enable_shared_from_this<Tensor> {

private:
    torch::Tensor _data;
    bool _requires_grad;
    Device _device;
    
    std::optional<std::string> _operation;
    mutable std::optional<torch::Tensor> _grad;
    std::optional<BackwardFunction> _grad_fn;

public:
    // Constructor
    Tensor(const torch::Tensor& tensor, bool requires_grad=false, Device device=Device::CPU(), std::optional<std::string> operation=std::nullopt);

    const torch::Tensor& data() const;
    const bool requires_grad() const;
    const Device device() const;

    const std::optional<const std::string> operation() const;
    const std::optional<const torch::Tensor>& grad() const;
    void set_grad(const torch::Tensor& grad) const;

    const std::optional<const BackwardFunction>& grad_fn() const;
    void set_grad_fn(const BackwardFunction& grad_fn);


    size_t numel() const;
    size_t dim() const;
    std::vector<int64_t> shape() const;

    void backward(const std::optional<const Tensor>& grad=std::nullopt);

    TensorPtr operator+(const TensorPtr& other);
    TensorPtr operator*(const TensorPtr& other);

    TensorPtr add(const TensorPtr& other);
    // Tensor mul(const Tensor& other) const;
    // Tensor matmul(const Tensor& other) const;

    std::string to_string() const;
};

} // namespace synapx

#endif // TENSOR_HPP
