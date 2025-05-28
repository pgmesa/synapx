#ifndef CPU_OPS_HPP
#define CPU_OPS_HPP

#include <synapx/autograd/engine.hpp>


namespace synapx::autograd::cpu {

    class Add: public Function {
    public:
        std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;
        std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& grad_outputs) override;
    
    private:
        torch::IntArrayRef shape_t1;
        torch::IntArrayRef shape_t2;
    };

    class Mul: public Function {
    public:
        std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;
        std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& grad_outputs) override;
    
    private:
        torch::Tensor t1;
        torch::Tensor t2;
    };

    class Matmul: public Function {
    public:
        std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;
        std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& grad_outputs) override;
    
    private:
        torch::Tensor t1;
        torch::Tensor t2;
    };

}


#endif