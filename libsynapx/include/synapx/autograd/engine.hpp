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

    SYNAPX_API void run_backward(const Tensor& tensor, const Tensor& grad);
    
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