
#include <synapx/autograd/functions.hpp>

#include <stdexcept>

#include <synapx/tensor.hpp>
#include <synapx/functional.hpp>
#include <synapx/autograd/utils.hpp>


namespace synapx::autograd
{
    constexpr int64_t epsilon = 1e-12;


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


    AddBackward0::AddBackward0(const Tensor& t1, const Tensor& t2)
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

    std::string AddBackward0::name() const { 
        return "AddBackward0"; 
    }

    TensorList AddBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_output = inputs[0];
        
        Tensor grad_t1, grad_t2;
        if (t1_req_grad) 
            grad_t1 = unbroadcast(grad_output, t1_shape);
        if (t2_req_grad) 
            grad_t2 = unbroadcast(grad_output, t2_shape);

        return {grad_t1, grad_t2};
    }


    SubBackward0::SubBackward0(const Tensor& t1, const Tensor& t2)
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

    std::string SubBackward0::name() const { 
        return "SubBackward0";
    }

    TensorList SubBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_output = inputs[0];
        
        Tensor grad_t1, grad_t2;
        if (t1_req_grad) 
            grad_t1 = unbroadcast(grad_output, t1_shape);
        if (t2_req_grad) 
            grad_t2 = -unbroadcast(grad_output, t2_shape);

        return {grad_t1, grad_t2};
    }


    MulBackward0::MulBackward0(const Tensor& t1, const Tensor& t2)
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

    std::string MulBackward0::name() const { 
        return "MulBackward0"; 
    }

    TensorList MulBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_output = inputs[0];
        
        Tensor grad_t1, grad_t2;
        if (t1_req_grad)
            grad_t1 = unbroadcast(grad_output * t2, t1_shape);
        
        if (t2_req_grad)
            grad_t2 = unbroadcast(grad_output * t1, t2_shape);

        return {grad_t1, grad_t2};
    }


    DivBackward0::DivBackward0(const Tensor& t1, const Tensor& t2)
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

    std::string DivBackward0::name() const { 
        return "DivBackward0"; 
    }

    TensorList DivBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_output = inputs[0];
        
        Tensor grad_t1, grad_t2;
        if (t1_req_grad)
            grad_t1 = unbroadcast(grad_output / t2, t1_shape);
        
        if (t2_req_grad)
            grad_t2 = unbroadcast((-grad_output * t1) / (t2 * t2), t2.sizes());

        return {grad_t1, grad_t2};
    }


    MatmulBackward0::MatmulBackward0(const Tensor& t1, const Tensor& t2)
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

    std::string MatmulBackward0::name() const { 
        return "MatmulBackward0"; 
    }

    TensorList MatmulBackward0::apply(const TensorList& inputs) {
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


    PowBackward0::PowBackward0(const Tensor& base, const Tensor& exp, const Tensor& fw_result)
        : base_req_grad(base.requires_grad()), exp_req_grad(exp.requires_grad()), base(base), fw_result(fw_result) {
        
        if (base_req_grad)
            this->exp = exp;
    }

    std::string PowBackward0::name() const { 
        return "PowBackward0"; 
    }

    TensorList PowBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_output = inputs[0];
        
        Tensor grad_base, grad_exp;

        if (base_req_grad)
            grad_base = exp * fw_result.div(base) * grad_output;
        
        if (exp_req_grad)
            grad_exp = fw_result * base.log() * grad_output;

        return {grad_base, grad_exp};
    }


    std::string CloneBackward0::name() const { 
        return "CloneBackward0"; 
    }

    TensorList CloneBackward0::apply(const TensorList& inputs) {
        return inputs;
    }


    AddmmBackward0::AddmmBackward0(const Tensor& inp, const Tensor& mat1, const Tensor& mat2)
        : inp_req_grad(inp.requires_grad()), mat1_req_grad(mat1.requires_grad()), mat2_req_grad(mat2.requires_grad()) {
        
        if (inp_req_grad) {
            inp_shape.reserve(inp.dim());
            inp_shape = inp.sizes().vec();
        }
        
        if (mat1_req_grad)
            this->mat2 = mat2;
        
        if (mat2_req_grad)
            this->mat1 = mat1;

    }

    std::string AddmmBackward0::name() const { 
        return "AddmmBackward0"; 
    }

    TensorList AddmmBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_output = inputs[0];
        Tensor grad_inp, grad_mat1, grad_mat2;

        if (inp_req_grad) 
            grad_inp = unbroadcast(grad_output, inp_shape);
        
        if (mat1_req_grad) 
            grad_mat1 = synapx::matmul(grad_output, synapx::swapdims(mat2, -2, -1));
        
        if (mat2_req_grad) 
            grad_mat2 = synapx::matmul(synapx::swapdims(mat1, -2, -1), grad_output);

        return {grad_inp, grad_mat1, grad_mat2};
    }


    ExpBackward0::ExpBackward0(const Tensor& t, const Tensor& fw_result) : t_req_grad(t.requires_grad()) {
        if (t_req_grad)
            this->fw_result = fw_result;
    }

    std::string ExpBackward0::name() const { 
        return "ExpBackward0"; 
    }

    TensorList ExpBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_output = inputs[0];
        return {fw_result * grad_output};
    }

    
    LogBackward0::LogBackward0(const Tensor& t) : t_req_grad(t.requires_grad()) {
        if (t_req_grad)
            this->t = t;
    }

    std::string LogBackward0::name() const { 
        return "LogBackward0"; 
    }

    TensorList LogBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_output = inputs[0];
        return {grad_output / (t + epsilon)};
    }
    

    SqrtBackward0::SqrtBackward0(const Tensor& t, const Tensor& fw_result) : t_req_grad(t.requires_grad()) {
        if (t_req_grad)
            this->fw_result = fw_result;
    }

    std::string SqrtBackward0::name() const { 
        return "SqrtBackward0"; 
    }

    TensorList SqrtBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_output = inputs[0];
        return {grad_output / (2 * fw_result)};
    }


    SumBackward0::SumBackward0(const Tensor& t, const torch::IntArrayRef& dim, bool keepdim)
        : t_req_grad(t.requires_grad()), dim(dim.vec()), keepdim(keepdim) {
        
        if (t_req_grad) {
            t_shape.reserve(t.dim());
            t_shape = t.sizes().vec();
        }
    }

    std::string SumBackward0::name() const { 
        return "SumBackward0"; 
    }

    TensorList SumBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_output = inputs[0];
        
        Tensor grad = grad_output;

        if(!keepdim && !dim.empty())
            grad = expand_dims(grad, dim);
        
        return {synapx::broadcast_to(grad, t_shape)};
    }


    MeanBackward0::MeanBackward0(const Tensor& t, const torch::IntArrayRef& dim, bool keepdim)
        : t_req_grad(t.requires_grad()), dim(dim.vec()), keepdim(keepdim) {
        
        if (t_req_grad) {
            t_shape.reserve(t.dim());
            t_shape = t.sizes().vec();

            normalized_dims = normalize_dims(t.dim(), dim);
        }
    }

    std::string MeanBackward0::name() const { 
        return "MeanBackward0"; 
    }

    TensorList MeanBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_output = inputs[0];
        
        Tensor grad = grad_output;

        if(!keepdim && !dim.empty())
            grad = expand_dims(grad, normalized_dims, /*normalized=*/true);
        
        int num_samples = 1;
        for (int64_t d : normalized_dims) {
            num_samples *= t_shape[d];
        }
        grad = synapx::broadcast_to(grad, t_shape) / num_samples;

        return {grad};
    }


    MaxBackward0::MaxBackward0(const Tensor& t, int64_t dim, bool keepdim, const Tensor& max_indices)
        : t_req_grad(t.requires_grad()), dim(dim), keepdim(keepdim), max_indices(max_indices) {
        
        if (t_req_grad) {
            t_shape.reserve(t.dim());
            t_shape = t.sizes().vec();
        }
    }

    std::string MaxBackward0::name() const { 
        return "MaxBackward0"; 
    }

    TensorList MaxBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_output = inputs[0];
        
        Tensor grad = grad_output;
        Tensor mask = synapx::zeros(t_shape, false, grad.options());

        // Along specific dimension
        Tensor indices = max_indices;
        if (!keepdim) {
            grad.unsqueeze_(dim);
            indices.unsqueeze_(dim);
        }
        mask.scatter_(dim, indices, 1);
        
        return {grad * mask};
    }

    MaxBackward1::MaxBackward1(const Tensor& t, const Tensor& max_value) : t(t), max_value(max_value) {}

    std::string MaxBackward1::name() const { 
        return "MaxBackward1";
    }

    TensorList MaxBackward1::apply(const TensorList& inputs) {
        const Tensor& grad_output = inputs[0];
        
        Tensor mask = (t == max_value).to(grad_output.dtype());
        mask /= mask.count_nonzero();
        
        return {grad_output * mask};
    }


    MinBackward0::MinBackward0(const Tensor& t, int64_t dim, bool keepdim, const Tensor& min_indices)
        : t_req_grad(t.requires_grad()), dim(dim), keepdim(keepdim), min_indices(min_indices) {
        
        if (t_req_grad) {
            t_shape.reserve(t.dim());
            t_shape = t.sizes().vec();
        }
    }

    std::string MinBackward0::name() const { 
        return "MinBackward0"; 
    }

    TensorList MinBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_output = inputs[0];
        
        Tensor grad = grad_output;
        Tensor mask = synapx::zeros(t_shape, false, grad.options());

        // Along specific dimension
        Tensor indices = min_indices;
        if (!keepdim) {
            grad.unsqueeze_(dim);
            indices.unsqueeze_(dim);
        }
        mask.scatter_(dim, indices, 1);
        
        return {grad * mask};
    }

    MinBackward1::MinBackward1(const Tensor& t, const Tensor& min_value) : t(t), min_value(min_value) {}

    std::string MinBackward1::name() const { 
        return "MinBackward1";
    }

    TensorList MinBackward1::apply(const TensorList& inputs) {
        const Tensor& grad_output = inputs[0];
        
        Tensor mask = (t == min_value).to(grad_output.dtype());
        mask /= mask.count_nonzero();
        
        return {grad_output * mask};
    }

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

