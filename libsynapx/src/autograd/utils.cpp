
#include <synapx/autograd/utils.hpp>

#include <torch/torch.h>

namespace synapx::autograd::utils {
    
    torch::Tensor unbroadcast(const torch::Tensor& grad, const std::vector<int64_t>& original_shape) {
        // Get the current shape of the gradient
        auto grad_shape = grad.sizes().vec();
        
        // If shapes are already the same, return a copy
        if (grad_shape == original_shape) {
            return grad.clone();
        }
        
        // Handle broadcasting - we need to sum across dimensions that were added
        torch::Tensor result = grad;
        
        // Sum across any dimensions that were broadcasted
        if (grad_shape.size() > original_shape.size()) {
            // Sum across leading dimensions that were added
            for (int i = 0; i < grad_shape.size() - original_shape.size(); i++) {
                result = result.sum(0, true);
            }
        }
        
        // Handle dimensions that were broadcasted from 1 to N
        for (int i = 0; i < result.dim(); i++) {
            int target_idx = i - (grad_shape.size() - original_shape.size());
            if (target_idx >= 0 && target_idx < original_shape.size() && original_shape[target_idx] == 1 && result.size(i) > 1) {
                result = result.sum(i, true);
            }
        }
        
        // Reshape to match original dimensions
        return result.reshape(original_shape);
    }


    torch::Tensor unbroadcast(const torch::Tensor& grad, torch::IntArrayRef& original_shape) {
        // Get the current shape of the gradient as an IntArrayRef
        auto grad_shape = grad.sizes();  // IntArrayRef

        // If shapes are already the same, just clone
        if (grad_shape.equals(original_shape)) {
            return grad.clone();
        }

        torch::Tensor result = grad;
        const auto grad_dim = static_cast<int64_t>(grad.dim());
        const auto orig_dim = static_cast<int64_t>(original_shape.size());

        // Sum across any extra leading dimensions that were added by broadcast
        if (grad_dim > orig_dim) {
            for (int64_t i = 0; i < grad_dim - orig_dim; ++i) {
                result = result.sum(0, /*keepdim=*/true);
            }
        }

        // Sum across dimensions that were broadcast from size 1 → N
        // We compute for each axis in the current tensor
        for (int64_t i = 0; i < result.dim(); ++i) {
            // Map this axis back to original_shape
            int64_t target_idx = i - (grad_dim - orig_dim);
            if (target_idx >= 0
                && target_idx < orig_dim
                && original_shape[target_idx] == 1
                && result.size(i) > 1
            ) {
                result = result.sum(i, /*keepdim=*/true);
            }
        }

        // Finally reshape back to exactly the original shape
        return result.reshape(original_shape);
    }

}