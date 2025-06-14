
#include <synapx/autograd/utils.hpp>

#include <torch/torch.h>
#include <synapx/tensor.hpp>


namespace synapx::autograd {

    synapx::Tensor unbroadcast(synapx::Tensor grad, torch::IntArrayRef original_shape) {
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

    synapx::Tensor expand_dims(synapx::Tensor tensor, torch::IntArrayRef dim, bool normalized) {
        int64_t k = tensor.dim();
        int64_t m = static_cast<int64_t>(dim.size());
        int64_t new_rank = k + m;
        
        torch::IntArrayRef normalized_dims = normalized? dim : normalize_dims(new_rank, dim);

        for (auto d : normalized_dims) {
            tensor = tensor.unsqueeze(d);
        }
        
        return tensor;
    }

    // Normalizes dimension indices. Converts negative indices to positive and sorts them.
    std::vector<int64_t> normalize_dims(int64_t tensor_dim, torch::IntArrayRef dim) {
        synapx::IntArray normalized;

        if (dim.empty()) {
            // All dimensions selected
            normalized.reserve(tensor_dim);
            for (int i = 0; i < tensor_dim; i++)
                normalized.push_back(i);
        } else {
            normalized.reserve(dim.size());
            for (auto d : dim) {
                int64_t d_norm = d;
                if (d_norm < 0) {
                    d_norm += tensor_dim;
                }
                normalized.push_back(d_norm);
            }

            std::sort(normalized.begin(), normalized.end());
        }

        return normalized;
    }


    /**
     * @brief Source: 
     * 
     * @param indices 
     * @param shape 
     * @return torch::Tensor 
     */
    torch::Tensor unravel_index(const torch::Tensor& indices, torch::IntArrayRef shape) {
        // Convert shape to tensor: (*shape, 1)
        std::vector<int64_t> shape_with_one(shape.begin(), shape.end());
        shape_with_one.push_back(1);
        torch::Tensor shape_tensor = torch::tensor(shape_with_one, indices.options());
        
        torch::Tensor coefs = shape_tensor.slice(0, 1).flip(0).cumprod(0).flip(0);
        
        // indices[..., None] - add dimension at the end
        torch::Tensor indices_expanded = indices.unsqueeze(-1);
        
        // torch.div(indices[..., None], coefs, rounding_mode='trunc') % shape[:-1]
        torch::Tensor shape_no_last = shape_tensor.slice(0, 0, -1);
        torch::Tensor result = torch::div(indices_expanded, coefs, /*rounding_mode=*/"trunc") % shape_no_last;
        
        return result;
    }

}