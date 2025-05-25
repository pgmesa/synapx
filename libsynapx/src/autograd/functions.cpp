
#include <synapx/autograd/functions.hpp>

#include <vector>

#include <synapx/autograd/utils.hpp>


namespace synapx::autograd {

    std::vector<torch::Tensor> Add::forward(const std::vector<torch::Tensor>& inputs) { 
        const torch::Tensor& t1 = inputs[0];
        const torch::Tensor& t2 = inputs[1];

        torch::Tensor out = torch::add(t1, t2);
        
        shape_t1 = t1.sizes();
        shape_t2 = t2.sizes();

        return {out};
    }

    std::vector<torch::Tensor> Add::backward(const std::vector<torch::Tensor>& grad_outputs) {
        torch::Tensor grad_t1 = utils::unbroadcast(grad_outputs[0], shape_t1);
        torch::Tensor grad_t2 = utils::unbroadcast(grad_outputs[0], shape_t2);
        return {grad_t1, grad_t2};
    }

}
