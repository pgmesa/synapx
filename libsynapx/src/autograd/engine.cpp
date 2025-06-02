
#include <synapx/autograd/engine.hpp>

#include <spdlog/spdlog.h>

#include <vector>
#include <stdexcept>


namespace synapx::autograd {

    void backward(std::shared_ptr<Function> grad_fn, const torch::Tensor& grad_output) {
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

}
