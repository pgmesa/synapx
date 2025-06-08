#ifndef CPU_OPS_HPP
#define CPU_OPS_HPP

#include <synapx/autograd/engine.hpp>

#include <optional>


namespace synapx::autograd::cpu {

    constexpr double epsilon = 1e-12;

    class Add: public Function {
    public:
        std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;
        std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& grad_outputs) override;
    
    private:
        std::vector<int64_t> t1_shape;
        std::vector<int64_t> t2_shape;
    };

    class Mul: public Function {
    public:
        std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;
        std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& grad_outputs) override;
    
    private:
        torch::Tensor t1;
        torch::Tensor t2;

        std::vector<int64_t> t1_shape;
        std::vector<int64_t> t2_shape;
    };

    class Div: public Function {
    public:
        std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;
        std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& grad_outputs) override;
    
    private:
        torch::Tensor t1;
        torch::Tensor t2;

        std::vector<int64_t> t1_shape;
    };

    class Matmul: public Function {
    public:
        std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;
        std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& grad_outputs) override;
    
    private:
        torch::Tensor t1;
        torch::Tensor t2;

        std::vector<int64_t> t1_shape;
        std::vector<int64_t> t2_shape;
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


    class Addmm: public Function {
    public:
        std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;
        std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& grad_outputs) override;
    
    private:
        std::vector<int64_t> inp_shape;
        torch::Tensor mat1;
        torch::Tensor mat2;
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
        torch::Tensor t;
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
       std::vector<int64_t> t_shape;
    };

    class Mean: public Function {
    public:
        Mean(const torch::IntArrayRef& dim = {}, bool keepdim = false);

        std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;
        std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& grad_outputs) override;
    
    private:
       const std::vector<int64_t> dim;
       const bool keepdim;
       std::vector<int64_t> t_shape;
       std::vector<int64_t> normalized_dims;
    };

    class Max: public Function {
    public:
        Max(std::optional<int64_t> dim = std::nullopt, bool keepdim = false);

        std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;
        std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& grad_outputs) override;
        
        torch::Tensor max_values;
        torch::Tensor max_indices;
    private:
       const std::optional<int64_t> dim;
       const bool keepdim;

       std::vector<int64_t> t_shape;
       torch::Tensor t;
    };

    class Min: public Function {
    public:
        Min(std::optional<int64_t> dim = std::nullopt, bool keepdim = false);

        std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;
        std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& grad_outputs) override;
        
        torch::Tensor min_values;
        torch::Tensor min_indices;
    private:
       const std::optional<int64_t> dim;
       const bool keepdim;

       std::vector<int64_t> t_shape;
       torch::Tensor t;
    };

    class Squeeze: public Function {
    public:
        Squeeze(const torch::IntArrayRef& dim = {});

        std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;
        std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& grad_outputs) override;
        
    private:
        std::vector<int64_t> dim;
        std::vector<int64_t> t_shape;
    };

    class Unsqueeze: public Function {
    public:
        Unsqueeze(int64_t dim);

        std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) override;
        std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& grad_outputs) override;
        
    private:
       const int64_t dim;
    };

}


#endif