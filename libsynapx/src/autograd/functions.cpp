
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
        const Tensor& grad_input = inputs[0];
        
        if (variable.grad().defined())
            variable.set_grad(variable.grad() + grad_input);
        else {
            variable.set_grad(grad_input);
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
        const Tensor& grad_input = inputs[0];
        
        Tensor grad_t1, grad_t2;
        if (t1_req_grad) 
            grad_t1 = unbroadcast(grad_input, t1_shape);
        if (t2_req_grad) 
            grad_t2 = unbroadcast(grad_input, t2_shape);

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
        const Tensor& grad_input = inputs[0];
        
        Tensor grad_t1, grad_t2;
        if (t1_req_grad) 
            grad_t1 = unbroadcast(grad_input, t1_shape);
        if (t2_req_grad) 
            grad_t2 = -unbroadcast(grad_input, t2_shape);

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
        const Tensor& grad_input = inputs[0];
        
        Tensor grad_t1, grad_t2;
        if (t1_req_grad)
            grad_t1 = unbroadcast(grad_input * t2, t1_shape);
        
        if (t2_req_grad)
            grad_t2 = unbroadcast(grad_input * t1, t2_shape);

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
        const Tensor& grad_input = inputs[0];
        
        Tensor grad_t1, grad_t2;
        if (t1_req_grad)
            grad_t1 = unbroadcast(grad_input / t2, t1_shape);
        
        if (t2_req_grad)
            grad_t2 = unbroadcast((-grad_input * t1) / (t2 * t2), t2.sizes());

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
        const Tensor& grad_input = inputs[0];
        
        Tensor grad_t1, grad_t2;
        if (t1_req_grad) {
            grad_t1 = synapx::matmul(grad_input, synapx::swapdims(t2, -2, -1));
            grad_t1 = unbroadcast(grad_t1, t1_shape);
        }

        if (t2_req_grad) {
            grad_t2 = synapx::matmul(synapx::swapdims(t1, -2, -1), grad_input);
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
        const Tensor& grad_input = inputs[0];
        
        Tensor grad_base, grad_exp;

        if (base_req_grad)
            grad_base = exp * fw_result.div(base) * grad_input;
        
        if (exp_req_grad)
            grad_exp = fw_result * base.log() * grad_input;

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
        const Tensor& grad_input = inputs[0];
        Tensor grad_inp, grad_mat1, grad_mat2;

        if (inp_req_grad) 
            grad_inp = unbroadcast(grad_input, inp_shape);
        
        if (mat1_req_grad) 
            grad_mat1 = synapx::matmul(grad_input, synapx::swapdims(mat2, -2, -1));
        
        if (mat2_req_grad) 
            grad_mat2 = synapx::matmul(synapx::swapdims(mat1, -2, -1), grad_input);

        return {grad_inp, grad_mat1, grad_mat2};
    }


    ExpBackward0::ExpBackward0(const Tensor& fw_result) : fw_result(fw_result) {}

    std::string ExpBackward0::name() const { 
        return "ExpBackward0"; 
    }

    TensorList ExpBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_input = inputs[0];
        return {fw_result * grad_input};
    }

    
    LogBackward0::LogBackward0(const Tensor& t) : t(t) {}

    std::string LogBackward0::name() const { 
        return "LogBackward0"; 
    }

    TensorList LogBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_input = inputs[0];
        return {grad_input / (t + epsilon)};
    }
    

    SqrtBackward0::SqrtBackward0(const Tensor& fw_result) : fw_result(fw_result) {}

    std::string SqrtBackward0::name() const { 
        return "SqrtBackward0"; 
    }

    TensorList SqrtBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_input = inputs[0];
        return {grad_input / (2 * fw_result)};
    }


    SumBackward0::SumBackward0(const Tensor& t, torch::IntArrayRef dim, bool keepdim)
        : t_shape(t.sizes().vec()), dim(dim.vec()), keepdim(keepdim) {}

    std::string SumBackward0::name() const { 
        return "SumBackward0"; 
    }

    TensorList SumBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_input = inputs[0];
        
        Tensor grad = grad_input;

        if(!keepdim && !dim.empty())
            grad = expand_dims(grad, dim);
        
        return {synapx::broadcast_to(grad, t_shape)};
    }


    MeanBackward0::MeanBackward0(const Tensor& t, torch::IntArrayRef dim, bool keepdim)
        : t_shape(t.sizes().vec()), dim(dim.vec()), keepdim(keepdim), normalized_dims(normalize_dims(t.dim(), dim)) {}

    std::string MeanBackward0::name() const { 
        return "MeanBackward0"; 
    }

    TensorList MeanBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_input = inputs[0];
        
        Tensor grad = grad_input;

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
        : t_shape(t.sizes().vec()), dim(dim), keepdim(keepdim), max_indices(max_indices) {}

    std::string MaxBackward0::name() const { 
        return "MaxBackward0"; 
    }

    TensorList MaxBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_input = inputs[0];
        
        Tensor grad = grad_input;
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
        const Tensor& grad_input = inputs[0];
        
        Tensor mask = (t == max_value).to(grad_input.dtype());
        mask /= mask.count_nonzero();
        
        return {grad_input * mask};
    }


    MinBackward0::MinBackward0(const Tensor& t, int64_t dim, bool keepdim, const Tensor& min_indices)
        : t_shape(t.sizes().vec()), dim(dim), keepdim(keepdim), min_indices(min_indices) {}

    std::string MinBackward0::name() const { 
        return "MinBackward0"; 
    }

    TensorList MinBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_input = inputs[0];
        
        Tensor grad = grad_input;
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
        const Tensor& grad_input = inputs[0];
        
        Tensor mask = (t == min_value).to(grad_input.dtype());
        mask /= mask.count_nonzero();
        
        return {grad_input * mask};
    }


    SqueezeBackward0::SqueezeBackward0(const Tensor& t, torch::IntArrayRef dim)
        : t_shape(t.sizes().vec()) {
            
        if (dim.empty()) {
            this->dim.reserve(t.dim());
            for (int i = 0; i < t.dim(); i++) {
                if (t.size(i) == 1)
                    this->dim.push_back(i);
            }
        } else {
            IntArray norm_dims = normalize_dims(t.dim(), dim);
            // Only add dimensions that have been squeezed
            for (int64_t d : dim) {
                if (t.size(d) == 1) {
                    this->dim.push_back(d); 
                }
            }
        }
    }

    std::string SqueezeBackward0::name() const { 
        return "SqueezeBackward0";
    }

    TensorList SqueezeBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_input = inputs[0];

        Tensor grad = expand_dims(grad_input, dim, true);
        
        return {grad};
    }


    UnsqueezeBackward0::UnsqueezeBackward0(int64_t dim) : dim(dim) {}

    std::string UnsqueezeBackward0::name() const { 
        return "UnsqueezeBackward0";
    }

    TensorList UnsqueezeBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_input = inputs[0];
        return {grad_input.squeeze(dim)};
    }


    ReshapeBackward0::ReshapeBackward0(const Tensor& t)
        : t_shape(t.sizes().vec()) {}

    std::string ReshapeBackward0::name() const { 
        return "ReshapeBackward0";
    }

    TensorList ReshapeBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_input = inputs[0];
        return {grad_input.reshape(t_shape)};
    }


    TransposeBackward0::TransposeBackward0(int64_t dim0, int64_t dim1): dim0(dim0), dim1(dim1) {}

    std::string TransposeBackward0::name() const { 
        return "TransposeBackward0";
    }

    TensorList TransposeBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_input = inputs[0];
        return {grad_input.transpose(dim0, dim1)};
    }


    MovedimBackward0::MovedimBackward0(int64_t src, int64_t dest): src(src), dest(dest) {}

    std::string MovedimBackward0::name() const { 
        return "MovedimBackward0";
    }

    TensorList MovedimBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_input = inputs[0];
        return {grad_input.movedim(src, dest)};
    }


    SliceBackward0::SliceBackward0(const Tensor& t, const TensorIndices& indices)
        : t_shape(t.sizes().vec()), indices(indices) {}

    std::string SliceBackward0::name() const { 
        return "SliceBackward0";
    }

    TensorList SliceBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_input = inputs[0];
        Tensor grad = synapx::zeros(t_shape, false, grad_input.options());
        grad.index_put_(indices, grad_input);
        return {grad};
    }


    ConcatBackward0::ConcatBackward0(const TensorList& inputs, int64_t dim) : dim(dim) {
        sizes.reserve(inputs.size());
        for (const auto& input : inputs)
            sizes.push_back(input.size(dim));
    }

    std::string ConcatBackward0::name() const { 
        return "ConcatBackward0";
    }

    TensorList ConcatBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_input = inputs[0];
        return synapx::split(grad_input, sizes, dim);
    }


    StackBackward0::StackBackward0(int64_t dim) : dim(dim) {}

    std::string StackBackward0::name() const { 
        return "StackBackward0";
    }

    TensorList StackBackward0::apply(const TensorList& inputs) {
        const Tensor& grad_input = inputs[0];
        return synapx::unbind(grad_input, dim);
    }


    UnbindBackward0::UnbindBackward0(const Tensor& t, int64_t dim) 
        : t_shape(t.sizes().vec()), dim(dim) {}

    std::string UnbindBackward0::name() const { 
        return "UnbindBackward0";
    }

    TensorList UnbindBackward0::apply(const TensorList& inputs) {
        // Handle negative dim
        int64_t actual_dim = dim < 0 ? t_shape.size() + dim : dim;
        
        // Create zero tensor and fill slices
        Tensor grad = synapx::zeros(t_shape, false, inputs[0].options());
        for (size_t inp_idx = 0; inp_idx < inputs.size(); inp_idx++) {
            const Tensor& input_grad = inputs[inp_idx];
            if (!input_grad.defined()) continue;
            grad.select(actual_dim, inp_idx).copy_(input_grad);
        }
        
        return {grad};
    }


    // Activations
    ReLUBackward0::ReLUBackward0(const Tensor& t) : t(t) {}

    std::string ReLUBackward0::name() const { 
        return "ReLUBackward0";
    }

    TensorList ReLUBackward0::apply(const TensorList& inputs) { 
        const Tensor& grad = inputs[0];
        return {grad.where(t > 0, 0)};
    }


    SigmoidBackward0::SigmoidBackward0(const Tensor& fw_result) : fw_result(fw_result) {}

    std::string SigmoidBackward0::name() const { 
        return "SigmoidBackward0";
    }

    TensorList SigmoidBackward0::apply(const TensorList& inputs) { 
        Tensor grad = inputs[0];
        return { grad * fw_result * (1 - fw_result) };
    }


    SoftmaxBackward0::SoftmaxBackward0(const Tensor& fw_result, int64_t dim) : fw_result(fw_result), dim(dim) {}

    std::string SoftmaxBackward0::name() const { 
        return "SoftmaxBackward0";
    }

    TensorList SoftmaxBackward0::apply(const TensorList& inputs) { 
        const Tensor& grad = inputs[0];
        
        Tensor softmax_grad_prod = fw_result * grad;
        Tensor sum_term = softmax_grad_prod.sum(dim, /*keepdim=*/true);
        Tensor input_grad = fw_result * (grad - sum_term);
        
        return {input_grad};
    }


    LogSoftmaxBackward0::LogSoftmaxBackward0(const Tensor& fw_result, int64_t dim) : fw_result(fw_result), dim(dim) {}

    std::string LogSoftmaxBackward0::name() const { 
        return "LogSoftmaxBackward0";
    }

    TensorList LogSoftmaxBackward0::apply(const TensorList& inputs) { 
        const Tensor& grad = inputs[0];

        Tensor grad_sum = grad.sum(dim, /*keepdim=*/true);
        Tensor softmax_output = fw_result.exp();
        Tensor input_grad = grad - softmax_output * grad_sum;
        
        return {input_grad};
    }


    // Losses
    MSELossBackward0::MSELossBackward0(const Tensor& input, const Tensor& target, const Tensor& diff, Reduction reduction) 
        : input_req_grad(input.requires_grad()), target_req_grad(target.requires_grad()), diff(diff), reduction(reduction) {}

    std::string MSELossBackward0::name() const { 
        return "MSELossBackward0";
    }

    TensorList MSELossBackward0::apply(const TensorList& inputs) { 
        const Tensor& grad = inputs[0];

        Tensor input_grad, target_grad;
        float scale = 2.0f;
        if (reduction == Reduction::Mean) {
            scale /= static_cast<float>(diff.numel());
        }

        if (input_req_grad) {
            input_grad = scale * diff * grad;
        }
        if (target_req_grad) {
            target_grad = -scale * diff * grad;
        }

        return {input_grad, target_grad};
    }

    NLLLossBackward0::NLLLossBackward0(const Tensor& input, const Tensor& target, Reduction reduction) 
        : input(input), target(target), reduction(reduction) {}

    std::string NLLLossBackward0::name() const { 
        return "NLLLossBackward0";
    }

    TensorList NLLLossBackward0::apply(const TensorList& inputs) { 
        const Tensor& grad = inputs[0];

        Tensor input_grad = synapx::zeros(input.sizes(), false, grad.options());
        Tensor target_grad;
        
        int64_t batch_size = input.size(0);
        int64_t num_classes = input.size(1);
        
        for (int64_t i = 0; i < batch_size; i++) {
            int64_t target_class = target[i].item().toInt();

            // Only update gradient for the target class
            if (target_class >= 0 && target_class < num_classes) {
                if (reduction == Reduction::Mean) {
                    input_grad.index_put_({i, target_class}, -grad / batch_size);
                } else if (reduction == Reduction::Sum) {
                    input_grad.index_put_({i, target_class}, -grad);
                } else {
                    input_grad.index_put_({i, target_class}, -grad[i]);
                }
            }
        }

        return {input_grad, target_grad};
    }

    
} // namespace synapx::autograd::functions

