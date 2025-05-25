#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include <synapx/autograd/engine.hpp>


namespace synapx::autograd {

    class Add: public Function {
    public:
        std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;
        std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& grad_outputs) override;
    
    private:
        torch::IntArrayRef shape_t1;
        torch::IntArrayRef shape_t2;
    };

}


#endif