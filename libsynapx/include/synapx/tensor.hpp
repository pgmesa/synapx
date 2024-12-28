#ifndef TENSOR_HPP
#define TENSOR_HPP


#include <torch/torch.h>
#include <cstddef>  // for size_t

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
    size_t ndim() const;
};

} // namespace synapx

#endif // TENSOR_HPP
