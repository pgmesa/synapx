#ifndef ENGINE_HPP
#define ENGINE_HPP

#include <stack>
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <memory>
#include <vector>

#include <torch/torch.h>
#include <synapx/core.hpp>
#include <synapx/tensor.hpp>


namespace synapx::autograd {
    class SYNAPX_API Function {
    public:
        std::vector<bool> requires_grad_flags;

        virtual std::string name() const = 0;
        virtual std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) = 0;
        virtual std::vector<torch::Tensor> backward(const torch::Tensor& output_grad, int output_idx) = 0;

        virtual ~Function() = default;
    };

    struct SYNAPX_API BackwardEdge {
        std::shared_ptr<BackwardNode> next_node;
        size_t input_slot;
        Tensor variable; // the Tensor/Variable to accumulate grad into
    };

    class SYNAPX_API BackwardNode {
    public:
        BackwardNode(std::shared_ptr<Function> fn, int output_idx, const std::vector<BackwardEdge>& backward_edges);
        
        std::string name() const;
        std::vector<BackwardEdge>& edges();
        inline std::vector<torch::Tensor> backward(const torch::Tensor& grad);
    
    private:
        std::shared_ptr<Function> fn;
        int output_idx;
        std::vector<BackwardEdge> backward_edges;

    };

    SYNAPX_API void backward(const synapx::Tensor& tensor, const torch::Tensor& grad);
    
    class SYNAPX_API AutogradState {
    public:
        static AutogradState& getInstance();
        
        bool is_grad_enabled() const;
        void push_grad_state(bool enabled);
        void pop_grad_state();
        void set_grad_enabled(bool enabled);

    private:
        AutogradState() = default;
        
        std::stack<bool> grad_enabled_stack_;
        
        AutogradState(const AutogradState&) = delete;
        AutogradState& operator=(const AutogradState&) = delete;
        AutogradState(AutogradState&&) = delete;
        AutogradState& operator=(AutogradState&&) = delete;
    };

    class SYNAPX_API NoGradGuard {
    public:
        NoGradGuard();
        ~NoGradGuard();
        
        NoGradGuard(const NoGradGuard&) = delete;
        NoGradGuard& operator=(const NoGradGuard&) = delete;
        NoGradGuard(NoGradGuard&&) = delete;
        NoGradGuard& operator=(NoGradGuard&&) = delete;

    private:
        bool prev_state_;
    };

    SYNAPX_API bool is_grad_enabled();
    SYNAPX_API void set_grad_enabled(bool enabled);

}

#endif