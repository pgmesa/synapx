#ifndef ENGINE_HPP
#define ENGINE_HPP


#include <torch/torch.h>
#include <synapx/tensor.hpp>


namespace synapx::autograd {

    class Function {
    public:
        struct BackwardEdge {
            std::shared_ptr<Function> next_fn;
            size_t input_slot;
            Tensor variable; // the Tensor/Variable to accumulate grad into
        };

        std::vector<BackwardEdge> backward_edges;

        virtual std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) = 0;
        virtual std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& grad_outputs) = 0;

        virtual ~Function() = default;
    };

    void backward(std::shared_ptr<Function> grad_fn, const torch::Tensor& grad_output);

}

#endif