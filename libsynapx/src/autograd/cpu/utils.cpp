
#include <synapx/autograd/cpu/utils.hpp>

#include <torch/torch.h>


namespace synapx::autograd::cpu {

    torch::Tensor unbroadcast(torch::Tensor grad, const torch::IntArrayRef& original_shape) {
        int64_t grad_dim = grad.dim();
        int64_t orig_dim = original_shape.size();

        // If grad has fewer dims, unsqueeze front dims to match
        if (grad_dim < orig_dim) {
            int64_t diff = orig_dim - grad_dim;
            for (int64_t i = 0; i < diff; ++i) {
                grad = grad.unsqueeze(0);
            }
            grad_dim = orig_dim;
        } else {
            // Sum extra leading dims if any
            if (grad_dim > orig_dim) {
                for (int64_t i = 0; i < grad_dim - orig_dim; ++i) {
                    grad = grad.sum(0);
                }
            }

            // Sum dims broadcasted from 1 -> N
            for (int64_t i = 0; i < orig_dim; ++i) {
                if (grad.size(i) != original_shape[i]) {
                    grad = grad.sum(i, /*keepdim=*/true);
                }
            }
        }

        return grad.reshape(original_shape);
    }


}