
#include <synapx/autograd/functions.hpp>

#include <stdexcept>

#include <synapx/tensor.hpp>
#include <synapx/functional.hpp>
#include <synapx/autograd/utils.hpp>


namespace synapx::autograd
{

    AccumulateGrad::AccumulateGrad(const Tensor& variable): variable(variable) {
        if (!variable.requires_grad())
            throw std::runtime_error("Internal Error: AccumulateGrad variable does not require grad");
    }

    std::string AccumulateGrad::name() const { 
        return "AccumulateGrad"; 
    }

    TensorList AccumulateGrad::apply(const TensorList& inputs) {
        const Tensor& grad_output = inputs[0];
        
        if (variable.grad().defined())
            variable.set_grad(variable.grad() + grad_output);
        else {
            variable.set_grad(grad_output);
        }

        return {};
    }


    AddBackward::AddBackward(const Tensor& t1, const Tensor& t2)
        : t1_req_grad(t1.requires_grad()), t2_req_grad(t2.requires_grad()) {

        if (t1_req_grad) {
            t1_shape.reserve(t1.dim());
            t1_shape = t1.shape();
        }
        if (t2_req_grad) {
            t2_shape.reserve(t2.dim());
            t2_shape = t2.shape();
        }
    }

    std::string AddBackward::name() const { 
        return "AddBackward"; 
    }

    TensorList AddBackward::apply(const TensorList& inputs) {
        const Tensor& grad_output = inputs[0];
        
        Tensor grad_t1, grad_t2;
        if (t1_req_grad) 
            grad_t1 = unbroadcast(grad_output, t1_shape);
        if (t2_req_grad) 
            grad_t2 = unbroadcast(grad_output, t2_shape);

        return {grad_t1, grad_t2};
    }


    SubBackward::SubBackward(const Tensor& t1, const Tensor& t2)
        : t1_req_grad(t1.requires_grad()), t2_req_grad(t2.requires_grad()) {
        if (t1_req_grad) {
            t1_shape.reserve(t1.dim());
            t1_shape = t1.shape();
        }
        if (t2_req_grad) {
            t2_shape.reserve(t2.dim());
            t2_shape = t2.shape();
        }
    }

    std::string SubBackward::name() const { 
        return "SubBackward";
    }

    TensorList SubBackward::apply(const TensorList& inputs) {
        const Tensor& grad_output = inputs[0];
        
        Tensor grad_t1, grad_t2;
        if (t1_req_grad) 
            grad_t1 = unbroadcast(grad_output, t1_shape);
        if (t2_req_grad) 
            grad_t2 = -unbroadcast(grad_output, t2_shape);

        return {grad_t1, grad_t2};
    }


    MulBackward::MulBackward(const Tensor& t1, const Tensor& t2)
        : t1_req_grad(t1.requires_grad()), t2_req_grad(t2.requires_grad()) {
        if (t1_req_grad) {
            this->t2 = t2;
            t1_shape.reserve(t1.dim());
            t1_shape = t1.shape();
        }
        
        if (t2_req_grad) {
            this->t1 = t1;
            t2_shape.reserve(t2.dim());
            t2_shape = t2.shape();
        }
    }

    std::string MulBackward::name() const { 
        return "MulBackward"; 
    }

    TensorList MulBackward::apply(const TensorList& inputs) {
        const Tensor& grad_output = inputs[0];
        
        Tensor grad_t1, grad_t2;
        if (t1_req_grad)
            grad_t1 = unbroadcast(grad_output * t2, t1_shape);
        
        if (t2_req_grad)
            grad_t2 = unbroadcast(grad_output * t1, t2_shape);

        return {grad_t1, grad_t2};
    }


    DivBackward::DivBackward(const Tensor& t1, const Tensor& t2)
        : t1_req_grad(t1.requires_grad()), t2_req_grad(t2.requires_grad()) {
        if (t1_req_grad) {
            this->t2 = t2;
            t1_shape.reserve(t1.dim());
            t1_shape = t1.shape();
        }
        
        if (t2_req_grad) {
            this->t1 = t1;
            this->t2 = t2;
        }
    }

    std::string DivBackward::name() const { 
        return "DivBackward"; 
    }

    TensorList DivBackward::apply(const TensorList& inputs) {
        const Tensor& grad_output = inputs[0];
        
        Tensor grad_t1, grad_t2;
        if (t1_req_grad)
            grad_t1 = unbroadcast(grad_output / t2, t1_shape);
        
        if (t2_req_grad)
            grad_t2 = unbroadcast((-grad_output * t1) / (t2 * t2), t2.sizes());

        return {grad_t1, grad_t2};
    }


    MatmulBackward::MatmulBackward(const Tensor& t1, const Tensor& t2)
        : t1_req_grad(t1.requires_grad()), t2_req_grad(t2.requires_grad()) {
        if (t1_req_grad){
            this->t2 = t2;
            t1_shape.reserve(t1.dim());
            t1_shape = t1.sizes().vec();
        }

        if (t2_req_grad) {
            this->t1 = t1;
            t2_shape.reserve(t2.dim());
            t2_shape = t2.sizes().vec();
        } 
    }

    std::string MatmulBackward::name() const { 
        return "MatmulBackward"; 
    }

    TensorList MatmulBackward::apply(const TensorList& inputs) {
        const Tensor& grad_output = inputs[0];
        
        Tensor grad_t1, grad_t2;
        if (t1_req_grad) {
            grad_t1 = synapx::matmul(grad_output, synapx::swapdims(t2, -2, -1));
            grad_t1 = unbroadcast(grad_t1, t1_shape);
        }

        if (t2_req_grad) {
            grad_t2 = synapx::matmul(synapx::swapdims(t1, -2, -1), grad_output);
            grad_t2 = unbroadcast(grad_t2, t2_shape);
        } 

        return {grad_t1, grad_t2};
    }


    PowBackward::PowBackward(const Tensor& base, const Tensor& exp, const Tensor& fw_result)
        : base_req_grad(base.requires_grad()), exp_req_grad(exp.requires_grad()), base(base), fw_result(fw_result) {
        
        if (base_req_grad)
            this->exp = exp;
    }

    std::string PowBackward::name() const { 
        return "PowBackward"; 
    }

    TensorList PowBackward::apply(const TensorList& inputs) {
        const Tensor& grad_output = inputs[0];
        
        Tensor grad_base, grad_exp;

        if (base_req_grad)
            grad_base = exp * fw_result.div(base) * grad_output;
        
        if (exp_req_grad)
            grad_exp = fw_result * base.log() * grad_output;

        return {grad_base, grad_exp};
    }


    // std::string Addmm::name() const { return "Addmm"; };

    // std::vector<torch::Tensor> Addmm::forward(const std::vector<torch::Tensor>& inputs) { 
    //     const torch::Tensor& inp = inputs[0];
    //     const torch::Tensor& mat1 = inputs[1];
    //     const torch::Tensor& mat2 = inputs[2];

    //     torch::Tensor out = torch::addmm(inp, mat1, mat2);
        
    //     if (t1_req_grad) {
    //         inp_shape.reserve(inp.dim());
    //         inp_shape = inp.sizes().vec();
    //     }
        
    //     if (t2_req_grad)
    //         this->mat2 = mat2;
        
    //     if (requires_grad_flags[2])
    //         this->mat1 = mat1;

    //     return {out};
    // }

    // std::vector<torch::Tensor> Addmm::backward(const torch::Tensor& grad_output, int output_idx) {
    //     torch::Tensor grad_inp, grad_mat1, grad_mat2;

    //     if (t1_req_grad) 
    //         grad_inp = unbroadcast(grad_output, inp_shape);
        
    //     if (t2_req_grad) 
    //         grad_mat1 = torch::matmul(grad_output, torch::swapdims(mat2, -2, -1));
        
    //     if (requires_grad_flags[2]) 
    //         grad_mat2 = torch::matmul(torch::swapdims(mat1, -2, -1), grad_output);

    //     return {grad_inp, grad_mat1, grad_mat2};
    // }


    // std::string Clone::name() const { return "Clone"; };

    // std::vector<torch::Tensor> Clone::forward(const std::vector<torch::Tensor>& inputs) { 
    //     return {inputs[0].clone()};
    // }

    // std::vector<torch::Tensor> Clone::backward(const torch::Tensor& grad_output, int output_idx) {
    //     return {grad_output};
    // }


    // std::string Exp::name() const { return "Exp"; };

    // std::vector<torch::Tensor> Exp::forward(const std::vector<torch::Tensor>& inputs) { 
    //     const torch::Tensor& t = inputs[0];
        
    //     torch::Tensor out = torch::exp(t);
        
    //     if (t1_req_grad)
    //         this->forward_result = out;
        
    //         return {out};
    // }

    // std::vector<torch::Tensor> Exp::backward(const torch::Tensor& grad_output, int output_idx) {
    //     return {forward_result * grad_output};
    // }


    // std::string Log::name() const { return "Log"; };

    // std::vector<torch::Tensor> Log::forward(const std::vector<torch::Tensor>& inputs) { 
    //     const torch::Tensor& t = inputs[0];
        
    //     torch::Tensor out = torch::log(t + epsilon);

    //     if (t1_req_grad)
    //        this->t = t;
        
    //     return {out};
    // }

    // std::vector<torch::Tensor> Log::backward(const torch::Tensor& grad_output, int output_idx) {
    //     return {grad_output / (t + epsilon)};
    // }


    // std::string Sqrt::name() const { return "Sqrt"; };

    // std::vector<torch::Tensor> Sqrt::forward(const std::vector<torch::Tensor>& inputs) { 
    //     const torch::Tensor& t = inputs[0];
        
    //     torch::Tensor out = torch::sqrt(t);
        
    //     if (t1_req_grad)
    //         this->forward_result = out;
        
    //     return {out};
    // }

    // std::vector<torch::Tensor> Sqrt::backward(const torch::Tensor& grad_output, int output_idx) {
    //     return {grad_output / (2 * forward_result)};
    // }

    SumBackward::SumBackward(const Tensor& t, const torch::IntArrayRef& dim, bool keepdim)
        : t_req_grad(t.requires_grad()), dim(dim.vec()), keepdim(keepdim) {
        
        if (t_req_grad) {
            t_shape.reserve(t.dim());
            t_shape = t.sizes().vec();
        }
    }

    std::string SumBackward::name() const { 
        return "SumBackward"; 
    }

    TensorList SumBackward::apply(const TensorList& inputs) {
        const Tensor& grad_output = inputs[0];
        
        Tensor grad = grad_output;

        if(!keepdim && !dim.empty())
            grad = expand_dims(grad, dim);
        
        return {synapx::broadcast_to(grad, t_shape)};
    }


    // Mean::Mean(const torch::IntArrayRef& dim, bool keepdim): dim(dim.vec()), keepdim(keepdim) {}

    // std::string Mean::name() const { return "Mean"; };

    // std::vector<torch::Tensor> Mean::forward(const std::vector<torch::Tensor>& inputs) {
    //     const torch::Tensor& t = inputs[0];

    //     if (t1_req_grad) {
    //         t_shape.reserve(t.dim());
    //         t_shape = t.sizes().vec();
            
    //         normalized_dims = normalize_dims(t.dim(), dim);
    //     }
        
    //     return {torch::mean(t, dim, keepdim)}; 
    // }

    // std::vector<torch::Tensor> Mean::backward(const torch::Tensor& grad_output, int output_idx) {
    //     torch::Tensor grad = grad_output;

    //     if(!keepdim && !dim.empty()) {
    //         grad = expand_dims(grad, normalized_dims, /*normalized=*/true);
    //     }

    //     int num_samples = 1;
    //     for (int64_t d : normalized_dims) {
    //         num_samples *= t_shape[d];
    //     }
    //     grad = torch::broadcast_to(grad, t_shape) / num_samples;

    //     return {grad};
    // }


    // Max::Max(std::optional<int64_t> dim, bool keepdim): dim(dim), keepdim(keepdim) {}

    // std::string Max::name() const { return "Max"; };

    // std::vector<torch::Tensor> Max::forward(const std::vector<torch::Tensor>& inputs) {
    //     const torch::Tensor& t = inputs[0];
        
    //     torch::Tensor result;
    //     if (!dim.has_value()) {
    //         result = torch::max(t);
    //         this->t = t;        
    //         max_values = result;
    //     } else {
    //         auto [values, indices] = torch::max(t, dim.value(), keepdim);
    //         result = values;
    //         max_indices = indices;
    //     }
        
    //     if (t1_req_grad) {
    //         t_shape.reserve(t.dim());
    //         t_shape = t.sizes().vec();
    //     }
        
    //     return {result};
    // }

    // std::vector<torch::Tensor> Max::backward(const torch::Tensor& grad_output, int output_idx) {
    //     torch::Tensor grad = grad_output;
    //     torch::Tensor mask;
    
    //     if (!dim.has_value()) {
    //         mask = (t == max_values).to(grad.dtype());
    //         mask /= mask.count_nonzero();
    //     } else {
    //         // Along specific dimension
    //         mask = torch::zeros(t_shape, grad.options());
    //         torch::Tensor indices = max_indices;
    //         if (!keepdim) {
    //             grad.unsqueeze_(dim.value());
    //             indices.unsqueeze_(dim.value());
    //         }
    //         mask.scatter_(dim.value(), indices, 1);
    //     }
        
    //     return {grad * mask};
    // }


    // Min::Min(std::optional<int64_t> dim, bool keepdim): dim(dim), keepdim(keepdim) {}

    // std::string Min::name() const { return "Min"; };

    // std::vector<torch::Tensor> Min::forward(const std::vector<torch::Tensor>& inputs) {
    //     const torch::Tensor& t = inputs[0];
        
    //     torch::Tensor result;
    //     if (!dim.has_value()) {
    //         result = torch::min(t);
    //         this->t = t;        
    //         min_values = result;
    //     } else {
    //         auto [values, indices] = torch::min(t, dim.value(), keepdim);
    //         result = values;
    //         min_indices = indices;
    //     }
        
    //     if (t1_req_grad) {
    //         t_shape.reserve(t.dim());
    //         t_shape = t.sizes().vec();
    //     }
        
    //     return {result};
    // }

    // std::vector<torch::Tensor> Min::backward(const torch::Tensor& grad_output, int output_idx) {
    //     torch::Tensor grad = grad_output;
    //     torch::Tensor mask;
    
    //     if (!dim.has_value()) {
    //         mask = (t == min_values).to(grad.dtype());
    //         mask /= mask.count_nonzero();
    //     } else {
    //         // Along specific dimension
    //         mask = torch::zeros(t_shape, grad.options());
    //         torch::Tensor indices = min_indices;
    //         if (!keepdim) {
    //             grad.unsqueeze_(dim.value());
    //             indices.unsqueeze_(dim.value());
    //         }
    //         mask.scatter_(dim.value(), indices, 1);
    //     }
        
    //     return {grad * mask};
    // }


    // Squeeze::Squeeze(const torch::IntArrayRef& dim): dim(dim.vec()) {}

    // std::string Squeeze::name() const { return "Squeeze"; };

    // std::vector<torch::Tensor> Squeeze::forward(const std::vector<torch::Tensor>& inputs) {
    //     const torch::Tensor& t = inputs[0];

    //     torch::Tensor result;
    //     if (dim.empty()) {
    //         result = t.squeeze();
    //         dim.reserve(t.dim());
    //         for (int i = 0; i < t.dim(); i++) {
    //             if (t.size(i) == 1)
    //                 dim.push_back(i);
    //         } 
    //     } else {
    //         result = t.squeeze(dim);
    //     }
        
    //     if (t1_req_grad) {
    //         t_shape.reserve(t.dim());
    //         t_shape = t.sizes().vec();
    //     }
        
    //     return {result};
    // }

    // std::vector<torch::Tensor> Squeeze::backward(const torch::Tensor& grad_output, int output_idx) {
    //     torch::Tensor grad = grad_output;
        
    //     grad = expand_dims(grad, dim).broadcast_to(t_shape);
        
    //     return {grad};
    // }


    // Unsqueeze::Unsqueeze(int64_t dim): dim(dim) {}

    // std::string Unsqueeze::name() const { return "Unsqueeze"; };

    // std::vector<torch::Tensor> Unsqueeze::forward(const std::vector<torch::Tensor>& inputs) {
    //     const torch::Tensor& t = inputs[0];

    //     torch::Tensor result = torch::unsqueeze(t, dim);
        
    //     return {result};
    // }

    // std::vector<torch::Tensor> Unsqueeze::backward(const torch::Tensor& grad_output, int output_idx) {
    //     torch::Tensor grad = grad_output;
        
    //     grad = grad.squeeze(dim);
        
    //     return {grad};
    // }


    // Reshape::Reshape(const torch::IntArrayRef& shape): shape(shape.vec()) {}

    // std::string Reshape::name() const { return "Reshape"; };

    // std::vector<torch::Tensor> Reshape::forward(const std::vector<torch::Tensor>& inputs) {
    //     const torch::Tensor& t = inputs[0];

    //     torch::Tensor result = torch::reshape(t, shape);

    //     if (t1_req_grad) {
    //         t_shape.reserve(t.dim());
    //         t_shape = t.sizes().vec();
    //     }
        
    //     return {result};
    // }

    // std::vector<torch::Tensor> Reshape::backward(const torch::Tensor& grad_output, int output_idx) {      
    //     return {grad_output.reshape(t_shape)};
    // }


    // Transpose::Transpose(int64_t dim0, int64_t dim1): dim0(dim0), dim1(dim1) {}

    // std::string Transpose::name() const { return "Transpose"; };

    // std::vector<torch::Tensor> Transpose::forward(const std::vector<torch::Tensor>& inputs) {
    //     const torch::Tensor& t = inputs[0];        
    //     return {torch::transpose(t, dim0, dim1)};
    // }

    // std::vector<torch::Tensor> Transpose::backward(const torch::Tensor& grad_output, int output_idx) {        
    //     return {grad_output.transpose(dim0, dim1)};
    // }


    // Movedim::Movedim(int64_t src, int64_t dest): src(src), dest(dest) {}

    // std::string Movedim::name() const { return "Movedim"; };

    // std::vector<torch::Tensor> Movedim::forward(const std::vector<torch::Tensor>& inputs) {
    //     const torch::Tensor& t = inputs[0];
    //     return {torch::movedim(t, src, dest)};
    // }

    // std::vector<torch::Tensor> Movedim::backward(const torch::Tensor& grad_output, int output_idx) {
    //     return {grad_output.movedim(src, dest)};
    // }


    // Slice::Slice(const std::vector<torch::indexing::TensorIndex>& idx) : indices(idx) {}

    // std::string Slice::name() const { return "Slice"; };

    // std::vector<torch::Tensor> Slice::forward(const std::vector<torch::Tensor>& inputs) {
    //     const torch::Tensor& input = inputs[0];
    //     t_shape = input.sizes().vec();
    //     return {input.index(indices)};
    // }

    // std::vector<torch::Tensor> Slice::backward(const torch::Tensor& grad_output, int output_idx) {
    //     torch::Tensor grad_input = torch::zeros(t_shape, grad_output.options());
    //     grad_input.index_put_(indices, grad_output);
    //     return {grad_input};
    // }


    // Concat::Concat(int64_t dim) : dim(dim) {}

    // std::string Concat::name() const { return "Concat"; };

    // std::vector<torch::Tensor> Concat::forward(const std::vector<torch::Tensor>& inputs) {
    //     torch::Tensor result = torch::concat(inputs, dim);

    //     sizes.clear();
    //     sizes.reserve(inputs.size());
    //     for (const auto& input : inputs)
    //         sizes.push_back(input.size(dim));

    //     return {result};
    // }

    // std::vector<torch::Tensor> Concat::backward(const torch::Tensor& grad_output, int output_idx) {
    //     return torch::split(grad_output, sizes, dim);
    // }


    // Stack::Stack(int64_t dim) : dim(dim) {}

    // std::string Stack::name() const { return "Stack"; };

    // std::vector<torch::Tensor> Stack::forward(const std::vector<torch::Tensor>& inputs) {        
    //     return {torch::stack(inputs, dim)};
    // }

    // std::vector<torch::Tensor> Stack::backward(const torch::Tensor& grad_output, int output_idx) {
    //     return torch::unbind(grad_output, dim);
    // }

    
    // Unbind::Unbind(int64_t dim) : dim(dim) {}

    // std::string Unbind::name() const { return "Unbind"; };

    // std::vector<torch::Tensor> Unbind::forward(const std::vector<torch::Tensor>& inputs) {
    //     const torch::Tensor& t = inputs[0];
    //     if (t1_req_grad) {
    //         t_shape.reserve(t.dim());
    //         t_shape = t.sizes().vec();
    //     }
    //     return torch::unbind(t, dim);
    // }

    // std::vector<torch::Tensor> Unbind::backward(const torch::Tensor& grad_output, int output_idx) {
    //     // Handle negative dim
    //     int64_t actual_dim = dim < 0 ? t_shape.size() + dim : dim;
        
    //     // Create zero tensor and fill slices
    //     torch::Tensor result = torch::zeros(t_shape, grad_output.options());
    //     result.select(actual_dim, output_idx).copy_(grad_output);
        
    //     return {result};
    // }

    
} // namespace synapx::autograd::functions

