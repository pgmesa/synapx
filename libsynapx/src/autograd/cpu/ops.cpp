
#include <synapx/autograd/cpu/ops.hpp>

#include <vector>

#include <spdlog/spdlog.h>

#include <synapx/autograd/cpu/utils.hpp>


namespace synapx::autograd::cpu {

    std::vector<torch::Tensor> Add::forward(const std::vector<torch::Tensor>& inputs) { 
        const torch::Tensor& t1 = inputs[0];
        const torch::Tensor& t2 = inputs[1];

        torch::Tensor out = torch::add(t1, t2);
        
        if (requires_grad_flags[0]) {
            t1_shape.reserve(t1.dim());
            t1_shape = t1.sizes().vec();
        }
        if (requires_grad_flags[1]) {
            t2_shape.reserve(t2.dim());
            t2_shape = t2.sizes().vec();
        }
        
        return {out};
    }

    std::vector<torch::Tensor> Add::backward(const std::vector<torch::Tensor>& grad_outputs) {
        const torch::Tensor& grad = grad_outputs[0];

        torch::Tensor grad_t1, grad_t2;
        if (requires_grad_flags[0]) grad_t1 = unbroadcast(grad, t1_shape);
        if (requires_grad_flags[1]) grad_t2 = unbroadcast(grad, t2_shape);

        return {grad_t1, grad_t2};
    }


    std::vector<torch::Tensor> Mul::forward(const std::vector<torch::Tensor>& inputs) { 
        const torch::Tensor& t1 = inputs[0];
        const torch::Tensor& t2 = inputs[1];

        torch::Tensor out = torch::mul(t1, t2);
        
        if (requires_grad_flags[0]) {
            this->t2 = t2;
            t1_shape.reserve(t1.dim());
            t1_shape = t1.sizes().vec();
        }
        
        if (requires_grad_flags[1]) {
            this->t1 = t1;
            t2_shape.reserve(t2.dim());
            t2_shape = t2.sizes().vec();
        }

        return {out};
    }

    std::vector<torch::Tensor> Mul::backward(const std::vector<torch::Tensor>& grad_outputs) {
        const torch::Tensor& grad = grad_outputs[0];

        torch::Tensor grad_t1, grad_t2;

        if (requires_grad_flags[0])
            grad_t1 = unbroadcast(grad * t2, t1_shape);
        
        if (requires_grad_flags[1])
            grad_t2 = unbroadcast(grad * t1, t2_shape);

        return {grad_t1, grad_t2};
    }


    std::vector<torch::Tensor> Div::forward(const std::vector<torch::Tensor>& inputs) { 
        const torch::Tensor& t1 = inputs[0];
        const torch::Tensor& t2 = inputs[1];

        torch::Tensor out = torch::div(t1, t2);
        
        if (requires_grad_flags[0]) {
            this->t2 = t2;
            t1_shape.reserve(t1.dim());
            t1_shape = t1.sizes().vec();
        }
        
        if (requires_grad_flags[1]) {
            this->t1 = t1;
            this->t2 = t2;
        }

        return {out};
    }

    std::vector<torch::Tensor> Div::backward(const std::vector<torch::Tensor>& grad_outputs) {
        const torch::Tensor& grad = grad_outputs[0];

        torch::Tensor grad_t1, grad_t2;

        if (requires_grad_flags[0])
            grad_t1 = unbroadcast(grad / t2, t1_shape);
        
        if (requires_grad_flags[1])
            grad_t2 = unbroadcast((-grad * t1) / (t2 * t2), t2.sizes());

        return {grad_t1, grad_t2};
    }


    std::vector<torch::Tensor> Matmul::forward(const std::vector<torch::Tensor>& inputs) { 
        const torch::Tensor& t1 = inputs[0];
        const torch::Tensor& t2 = inputs[1];

        torch::Tensor out = torch::matmul(t1, t2);

        if (requires_grad_flags[0]){
            this->t2 = t2;
            t1_shape.reserve(t1.dim());
            t1_shape = t1.sizes().vec();
        }

        if (requires_grad_flags[1]) {
            this->t1 = t1;
            t2_shape.reserve(t2.dim());
            t2_shape = t2.sizes().vec();
        } 

        return {out};
    }

    std::vector<torch::Tensor> Matmul::backward(const std::vector<torch::Tensor>& grad_outputs) {
        const torch::Tensor& grad = grad_outputs[0];
        
        torch::Tensor grad_t1, grad_t2;

        if (requires_grad_flags[0]) {
            grad_t1 = torch::matmul(grad, torch::swapdims(t2, -2, -1));
            grad_t1 = unbroadcast(grad_t1, t1_shape);
        }

        if (requires_grad_flags[1]) {
            grad_t2 = torch::matmul(torch::swapdims(t1, -2, -1), grad);
            grad_t2 = unbroadcast(grad_t2, t2_shape);
        } 

        return {grad_t1, grad_t2};
    }

    std::vector<torch::Tensor> Pow::forward(const std::vector<torch::Tensor>& inputs) { 
        const torch::Tensor& base = inputs[0];
        const torch::Tensor& exp = inputs[1];

        torch::Tensor out = torch::pow(base, exp);

        this->base = base;
        this->forward_result = out;
        if (requires_grad_flags[0])
            this->exp = exp;

        return {out};
    }

    std::vector<torch::Tensor> Pow::backward(const std::vector<torch::Tensor>& grad_outputs) {
        const torch::Tensor& grad = grad_outputs[0];

        torch::Tensor grad_base, grad_exp;

        if (requires_grad_flags[0])
            grad_base = exp * forward_result.div(base) * grad;
        
        if (requires_grad_flags[1])
            grad_exp = forward_result * base.log() * grad;
        
        return {grad_base, grad_exp};
    }


    std::vector<torch::Tensor> Addmm::forward(const std::vector<torch::Tensor>& inputs) { 
        const torch::Tensor& inp = inputs[0];
        const torch::Tensor& mat1 = inputs[1];
        const torch::Tensor& mat2 = inputs[2];

        torch::Tensor out = torch::addmm(inp, mat1, mat2);
        
        if (requires_grad_flags[0]) {
            inp_shape.reserve(inp.dim());
            inp_shape = inp.sizes().vec();
        }
        
        if (requires_grad_flags[1])
            this->mat2 = mat2;
        
        if (requires_grad_flags[2])
            this->mat1 = mat1;

        return {out};
    }

    std::vector<torch::Tensor> Addmm::backward(const std::vector<torch::Tensor>& grad_outputs) {
        const torch::Tensor& grad = grad_outputs[0];
        
        torch::Tensor grad_inp, grad_mat1, grad_mat2;

        if (requires_grad_flags[0]) 
            grad_inp = unbroadcast(grad, inp_shape);
        
        if (requires_grad_flags[1]) 
            grad_mat1 = torch::matmul(grad, torch::swapdims(mat2, -2, -1));
        
        if (requires_grad_flags[2]) 
            grad_mat2 = torch::matmul(torch::swapdims(mat1, -2, -1), grad);

        return {grad_inp, grad_mat1, grad_mat2};
    }

    std::vector<torch::Tensor> Clone::forward(const std::vector<torch::Tensor>& inputs) { 
        return {inputs[0].clone()};
    }

    std::vector<torch::Tensor> Clone::backward(const std::vector<torch::Tensor>& grad_outputs) {
        return grad_outputs;
    }


    std::vector<torch::Tensor> Exp::forward(const std::vector<torch::Tensor>& inputs) { 
        const torch::Tensor& t = inputs[0];
        
        torch::Tensor out = torch::exp(t);
        
        if (requires_grad_flags[0])
            this->forward_result = out;
        
            return {out};
    }

    std::vector<torch::Tensor> Exp::backward(const std::vector<torch::Tensor>& grad_outputs) {
        return {forward_result * grad_outputs[0]};
    }


    std::vector<torch::Tensor> Log::forward(const std::vector<torch::Tensor>& inputs) { 
        const torch::Tensor& t = inputs[0];
        
        torch::Tensor out = torch::log(t + epsilon);

        if (requires_grad_flags[0])
           this->t = t;
        
        return {out};
    }

    std::vector<torch::Tensor> Log::backward(const std::vector<torch::Tensor>& grad_outputs) {
        return {grad_outputs[0] / (t + epsilon)};
    }


    std::vector<torch::Tensor> Sqrt::forward(const std::vector<torch::Tensor>& inputs) { 
        const torch::Tensor& t = inputs[0];
        
        torch::Tensor out = torch::sqrt(t);
        
        if (requires_grad_flags[0])
            this->forward_result = out;
        
        return {out};
    }

    std::vector<torch::Tensor> Sqrt::backward(const std::vector<torch::Tensor>& grad_outputs) {
        return {grad_outputs[0] / (2 * forward_result)};
    }


    Sum::Sum(const torch::IntArrayRef& dim, bool keepdim): dim(dim.vec()), keepdim(keepdim) {}

    std::vector<torch::Tensor> Sum::forward(const std::vector<torch::Tensor>& inputs) {
        const torch::Tensor& t = inputs[0];

        if (requires_grad_flags[0]) {
            t_shape.reserve(t.dim());
            t_shape = t.sizes().vec();
        }
        
        return {torch::sum(t, dim, keepdim)}; 
    }

    std::vector<torch::Tensor> Sum::backward(const std::vector<torch::Tensor>& grad_outputs) {
        torch::Tensor grad = grad_outputs[0];

        if(!keepdim && !dim.empty())
            grad = expand_dims(grad, dim);
        
        return {torch::broadcast_to(grad, t_shape)};
    }


    Mean::Mean(const torch::IntArrayRef& dim, bool keepdim): dim(dim.vec()), keepdim(keepdim) {}

    std::vector<torch::Tensor> Mean::forward(const std::vector<torch::Tensor>& inputs) {
        const torch::Tensor& t = inputs[0];

        if (requires_grad_flags[0]) {
            t_shape.reserve(t.dim());
            t_shape = t.sizes().vec();
            
            normalized_dims = normalize_dims(t.dim(), dim);
        }
        
        return {torch::mean(t, dim, keepdim)}; 
    }

    std::vector<torch::Tensor> Mean::backward(const std::vector<torch::Tensor>& grad_outputs) {
        torch::Tensor grad = grad_outputs[0];

        if(!keepdim && !dim.empty()) {
            grad = expand_dims(grad, normalized_dims, /*normalized=*/true);
        }

        int num_samples = 1;
        for (int64_t d : normalized_dims) {
            num_samples *= t_shape[d];
        }
        grad = torch::broadcast_to(grad, t_shape) / num_samples;

        return {grad};
    }
    
}
