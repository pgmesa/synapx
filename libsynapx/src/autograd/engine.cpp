
#include <synapx/autograd/engine.hpp>

#include <spdlog/spdlog.h>

#include <vector>
#include <stdexcept>


namespace synapx::autograd {

    void backward(const synapx::Tensor& tensor, const torch::Tensor& grad) {
        spdlog::debug("Backward called");

        if (!tensor.defined())
            throw std::runtime_error("Tensor passed to compute backward pass is not defined");

        std::shared_ptr<Function> grad_fn = tensor.grad_fn();

        if (!grad_fn) {
            throw std::runtime_error("No backward function defined for this tensor");
        }

        // Validate grad_output shape
        torch::Tensor grad_output = grad;
        if (!grad.defined()) {
            if (tensor.numel() == 1) 
                grad_output = torch::ones_like(tensor.data());
            else
                throw std::runtime_error("Grad can be implicitly created only for scalar outputs");
        }

        if (!grad_output.sizes().equals(tensor.data().sizes())) {
            throw std::runtime_error("Shape mismatch between input gradient and tensor");
        }

        // 1) Build topo order
        std::vector<std::shared_ptr<Function>> topo;
        std::unordered_set<Function*> seen;
        std::function<void(const std::shared_ptr<Function>&)> dfs =
            [&](auto fn) {
                if (!fn || seen.count(fn.get())) return;
                seen.insert(fn.get());
                
                for (auto& edge : fn->backward_edges) {
                    dfs(edge.next_fn);
                }
                topo.push_back(fn);
            };
        dfs(grad_fn);

        // 2) Initialize grad map
        std::unordered_map<Function*, torch::Tensor> grad_map;
        grad_map[grad_fn.get()] = grad_output;

        // 3) Walk backwards
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            auto current_fn = *it;
            auto grad_out_tensor = grad_map[current_fn.get()];

            // backward() returns one gradient per forward‐input
            auto grad_inputs = current_fn->backward({ grad_out_tensor });

            // dispatch each gradient along its edge
            for (auto& edge : current_fn->backward_edges) {
                size_t input_idx = edge.input_slot;
                if (input_idx >= grad_inputs.size()) {
                    throw std::runtime_error(
                        "autograd: backward() returned " +
                        std::to_string(grad_inputs.size()) +
                        " gradients, but edge.input_slot=" +
                        std::to_string(input_idx));
                }

                Tensor in_tensor = edge.variable;
                torch::Tensor grad_input = grad_inputs[input_idx];
                if (!in_tensor.requires_grad())
                    continue;

                if (edge.variable.is_leaf() || edge.variable.retains_grad()) {
                    // Leaf tensors always accumulate gradients
                    if (edge.variable.grad().defined()) {
                        edge.variable.set_grad(edge.variable.grad() + grad_input);
                    } else {
                        edge.variable.set_grad(grad_input);
                    }
                }
                // Also propagate to next function in computation graph
                auto next_fn = edge.variable.grad_fn();
                if (next_fn) {
                    auto& accumulated = grad_map[next_fn.get()];
                    if (accumulated.defined()) {
                        accumulated = accumulated + grad_input;
                    } else {
                        accumulated = grad_input;
                    }
                }
            }
        }
    }

    AutogradState& AutogradState::getInstance() {
        // Thread-local singleton for thread safety
        thread_local static AutogradState instance;
        return instance;
    }

    bool AutogradState::is_grad_enabled() const {
        return grad_enabled_stack_.empty() ? true : grad_enabled_stack_.top();
    }

    void AutogradState::push_grad_state(bool enabled) {
        grad_enabled_stack_.push(enabled);
    }

    void AutogradState::pop_grad_state() {
        if (!grad_enabled_stack_.empty()) {
            grad_enabled_stack_.pop();
        }
    }

    void AutogradState::set_grad_enabled(bool enabled) {
        if (grad_enabled_stack_.empty()) {
            grad_enabled_stack_.push(enabled);
        } else {
            grad_enabled_stack_.top() = enabled;
        }
    }

    NoGradGuard::NoGradGuard() : prev_state_(AutogradState::getInstance().is_grad_enabled()) {
        AutogradState::getInstance().push_grad_state(false);
    }

    NoGradGuard::~NoGradGuard() {
        AutogradState::getInstance().pop_grad_state();
    }

    bool is_grad_enabled() {
        return AutogradState::getInstance().is_grad_enabled();
    }

    void set_grad_enabled(bool enabled) {
        AutogradState::getInstance().set_grad_enabled(enabled);
    }

}
