
#include <synapx/autograd/cpu/ops.hpp>

#include <vector>

#include <synapx/autograd/cpu/utils.hpp>


namespace synapx::autograd::cpu {

    std::vector<torch::Tensor> Add::forward(const std::vector<torch::Tensor>& inputs) { 
        const torch::Tensor& t1 = inputs[0];
        const torch::Tensor& t2 = inputs[1];

        torch::Tensor out = torch::add(t1, t2);
        
        shape_t1 = t1.sizes();
        shape_t2 = t2.sizes();

        return {out};
    }

    std::vector<torch::Tensor> Add::backward(const std::vector<torch::Tensor>& grad_outputs) {
        torch::Tensor grad_t1 = unbroadcast(grad_outputs[0], shape_t1);
        torch::Tensor grad_t2 = unbroadcast(grad_outputs[0], shape_t2);
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
        torch::Tensor grad_t1 = unbroadcast(grad_outputs[0] * t2, t1.sizes());
        torch::Tensor grad_t2 = unbroadcast(grad_outputs[0] * t1, t2.sizes());
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
        torch::Tensor grad_t1 = torch::matmul(grad_outputs[0], torch::swapdims(t2, -2, -1));
        torch::Tensor grad_t2 = torch::matmul(torch::swapdims(t1, -2, -1), grad_outputs[0]);

        return {unbroadcast(grad_t1, t1.sizes()), unbroadcast(grad_t2, t2.sizes())};
    }

}
