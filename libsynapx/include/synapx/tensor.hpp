#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <cstddef>  // for size_t

#include <torch/torch.h>


#if defined(_WIN32) || defined(_WIN64)
#define SYNAPX_API __declspec(dllexport)
#else
#define SYNAPX_API __attribute__((visibility("default")))
#endif

namespace synapx {

class SYNAPX_API Tensor {
public:
    torch::Tensor data;
    
    // Constructor
    Tensor(const torch::Tensor& tensor);

    // Member functions
    size_t numel() const;
    size_t dim() const;
    std::vector<int64_t> shape() const;

    Tensor operator+(const Tensor& other);
    Tensor operator*(const Tensor& other);

    Tensor add(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;
};

} // namespace synapx

#endif // TENSOR_HPP
