#ifndef CPU_OPS_HPP
#define CPU_OPS_HPP

#include <synapx/autograd/engine.hpp>


namespace synapx::autograd::cpu {

    constexpr double epsilon = 1e-12;

    class Add: public Function {
    public:
        std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;
        std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& grad_outputs) override;
    
    private:
        std::vector<int64_t> shape_t1;
        std::vector<int64_t> shape_t2;
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

    class Addmm: public Function {
    public:
        std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;
        std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& grad_outputs) override;
    
    private:
        torch::Tensor inp;
        torch::Tensor mat1;
        torch::Tensor mat2;
    };

    class Pow: public Function {
    public:
        std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;
        std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& grad_outputs) override;
    
    private:
        torch::Tensor base;
        torch::Tensor exp;
        torch::Tensor forward_result;
    };

    class Clone: public Function {
    public:
        std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;
        std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& grad_outputs) override;
    };

    class Exp: public Function {
    public:
        std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;
        std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& grad_outputs) override;
    
    private:
        torch::Tensor forward_result;
    };

    class Log: public Function {
    public:
        std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;
        std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& grad_outputs) override;
    
    private:
        torch::Tensor t1;
    };

    class Sqrt: public Function {
    public:
        std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;
        std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& grad_outputs) override;
    
    private:
        torch::Tensor forward_result;
    };

    class Sum: public Function {
    public:
        Sum(const torch::IntArrayRef& dim = {}, bool keepdim = false);

        std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;
        std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& grad_outputs) override;
    
    private:
       const std::vector<int64_t> dim;
       const bool keepdim;
       std::vector<int64_t> t1_shape;
    };

}


#endif