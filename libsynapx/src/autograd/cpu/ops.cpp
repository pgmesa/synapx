
#include <synapx/autograd/cpu/ops.hpp>

#include <vector>

#include <spdlog/spdlog.h>

#include <synapx/autograd/cpu/utils.hpp>


namespace synapx::autograd::cpu {

    std::vector<torch::Tensor> Add::forward(const std::vector<torch::Tensor>& inputs) { 
        const torch::Tensor& t1 = inputs[0];
        const torch::Tensor& t2 = inputs[1];

        torch::Tensor out = torch::add(t1, t2);
        
        shape_t1.reserve(t1.dim());
        shape_t1 = t1.sizes().vec();
        shape_t2.reserve(t2.dim());
        shape_t2 = t2.sizes().vec();
        return {out};
    }

    std::vector<torch::Tensor> Add::backward(const std::vector<torch::Tensor>& grad_outputs) {
        const torch::Tensor& grad = grad_outputs[0];
        torch::Tensor grad_t1 = unbroadcast(grad, shape_t1);
        torch::Tensor grad_t2 = unbroadcast(grad, shape_t2);
        return {grad_t1, grad_t2};
    }


    std::vector<torch::Tensor> Mul::forward(const std::vector<torch::Tensor>& inputs) { 
        const torch::Tensor& t1 = inputs[0];
        const torch::Tensor& t2 = inputs[1];

        torch::Tensor out = torch::mul(t1, t2);
        
        this->t1 = t1;
        this->t2 = t2;

        return {out};
    }

    std::vector<torch::Tensor> Mul::backward(const std::vector<torch::Tensor>& grad_outputs) {
        const torch::Tensor& grad = grad_outputs[0];
        torch::Tensor grad_t1 = unbroadcast(grad * t2, t1.sizes());
        torch::Tensor grad_t2 = unbroadcast(grad * t1, t2.sizes());
        return {grad_t1, grad_t2};
    }


    std::vector<torch::Tensor> Matmul::forward(const std::vector<torch::Tensor>& inputs) { 
        const torch::Tensor& t1 = inputs[0];
        const torch::Tensor& t2 = inputs[1];

        torch::Tensor out = torch::matmul(t1, t2);
        
        this->t1 = t1;
        this->t2 = t2;

        return {out};
    }

    std::vector<torch::Tensor> Matmul::backward(const std::vector<torch::Tensor>& grad_outputs) {
        const torch::Tensor& grad = grad_outputs[0];
        torch::Tensor grad_t1 = torch::matmul(grad, torch::swapdims(t2, -2, -1));
        torch::Tensor grad_t2 = torch::matmul(torch::swapdims(t1, -2, -1), grad);

        return {unbroadcast(grad_t1, t1.sizes()), unbroadcast(grad_t2, t2.sizes())};
    }


    std::vector<torch::Tensor> Addmm::forward(const std::vector<torch::Tensor>& inputs) { 
        const torch::Tensor& inp = inputs[0];
        const torch::Tensor& mat1 = inputs[1];
        const torch::Tensor& mat2 = inputs[2];

        torch::Tensor out = torch::addmm(inp, mat1, mat2);
        
        this->inp = inp;
        this->mat1 = mat1;
        this->mat2 = mat2;

        return {out};
    }

    std::vector<torch::Tensor> Addmm::backward(const std::vector<torch::Tensor>& grad_outputs) {
        const torch::Tensor& grad = grad_outputs[0];
        
        torch::Tensor grad_inp = unbroadcast(grad, inp.sizes());
        torch::Tensor grad_mat1 = torch::matmul(grad, torch::swapdims(mat2, -2, -1));
        torch::Tensor grad_mat2 = torch::matmul(torch::swapdims(mat1, -2, -1), grad);

        return {grad_inp, grad_mat1, grad_mat2};
    }


    std::vector<torch::Tensor> Pow::forward(const std::vector<torch::Tensor>& inputs) { 
        const torch::Tensor& t1 = inputs[0];
        const torch::Tensor& exp = inputs[1];

        torch::Tensor out = torch::pow(t1, exp);
        
        this->t1 = t1;
        this->exp = exp;

        return {out};
    }

    std::vector<torch::Tensor> Pow::backward(const std::vector<torch::Tensor>& grad_outputs) {
        return {exp * (t1.pow(exp - 1)) * grad_outputs[0]};
    }


    std::vector<torch::Tensor> Clone::forward(const std::vector<torch::Tensor>& inputs) { 
        return {inputs[0].clone()};
    }

    std::vector<torch::Tensor> Clone::backward(const std::vector<torch::Tensor>& grad_outputs) {
        return grad_outputs;
    }


    std::vector<torch::Tensor> Exp::forward(const std::vector<torch::Tensor>& inputs) { 
        const torch::Tensor& t1 = inputs[0];
        torch::Tensor out = torch::exp(t1);
        this->forward_result = out;
        return {out};
    }

    std::vector<torch::Tensor> Exp::backward(const std::vector<torch::Tensor>& grad_outputs) {
        return {forward_result * grad_outputs[0]};
    }


    std::vector<torch::Tensor> Log::forward(const std::vector<torch::Tensor>& inputs) { 
        const torch::Tensor& t1 = inputs[0];
        torch::Tensor out = torch::log(t1 + epsilon);
        this->t1 = t1;
        return {out};
    }

    std::vector<torch::Tensor> Log::backward(const std::vector<torch::Tensor>& grad_outputs) {
        return {grad_outputs[0] / (t1 + epsilon)};
    }


    std::vector<torch::Tensor> Sqrt::forward(const std::vector<torch::Tensor>& inputs) { 
        const torch::Tensor& t1 = inputs[0];
        torch::Tensor out = torch::sqrt(t1);
        this->forward_result = out;
        return {out};
    }

    std::vector<torch::Tensor> Sqrt::backward(const std::vector<torch::Tensor>& grad_outputs) {
        return {grad_outputs[0] / (2 * forward_result)};
    }


    Sum::Sum(const torch::IntArrayRef& dim, bool keepdim): dim(dim.vec()), keepdim(keepdim) {}

    std::vector<torch::Tensor> Sum::forward(const std::vector<torch::Tensor>& inputs) {
        const torch::Tensor& t1 = inputs[0];
        t1_shape.reserve(t1.dim());
        t1_shape = t1.sizes().vec();
        return {torch::sum(t1, dim, keepdim)}; 
    }

    std::vector<torch::Tensor> Sum::backward(const std::vector<torch::Tensor>& grad_outputs) {
        torch::Tensor grad = grad_outputs[0];

        if(!keepdim && !dim.empty())
            grad = expand_dims(grad, dim);
        
        return {torch::broadcast_to(grad, t1_shape)};
    }
    
}
