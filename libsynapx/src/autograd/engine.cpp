
#include <synapx/autograd/engine.hpp>

#include <vector>
#include <stdexcept>

#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include <synapx/tensor.hpp>
#include <synapx/functional.hpp>
#include <synapx/autograd/graph.hpp>


namespace synapx::autograd {


    void run_backward(const Tensor& tensor, const Tensor& grad) {
        spdlog::debug("Backward called");

        if (!tensor.defined())
            throw std::runtime_error("Tensor passed to compute backward pass is not defined");

        NodePtr grad_fn = tensor.grad_fn();

        if (!grad_fn) {
            throw std::runtime_error("No backward function defined for this tensor");
        }

        // Validate grad_output shape
        Tensor grad_output = grad;
        if (!grad.defined()) {
            if (tensor.numel() == 1) 
                grad_output = synapx::ones_like(tensor);
            else
                throw std::runtime_error("Grad can be implicitly created only for scalar outputs");
        }

        if (!grad_output.sizes().equals(tensor.sizes())) {
            throw std::runtime_error("Shape mismatch between input gradient and tensor");
        }

        // 1) Build topological order
        std::vector<NodePtr> topo;
        std::unordered_set<NodePtr> seen;
        std::unordered_map<autograd::Node*, TensorList> grad_map;
        std::function<void(NodePtr)> dfs = [&](NodePtr grad_fn) {
            if (!grad_fn || seen.count(grad_fn)) return;
            seen.insert(grad_fn);

            TensorList grad_slots(grad_fn->num_inputs(), Tensor{});
            grad_map[grad_fn.get()] = std::move(grad_slots);
            
            for (const Edge& edge : grad_fn->get_next_edges()) {
                dfs(edge.node);
            }
            topo.push_back(grad_fn);
        };
        dfs(grad_fn);

        grad_map[grad_fn.get()][tensor.output_nr()] = grad_output;

        // 2) Walk backwards
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            NodePtr current_node = *it;
            TensorList input_grads = grad_map[current_node.get()];

            TensorList output_grads = current_node->apply(input_grads);

            // Dispatch each gradient along its edge
            EdgeList next_edges = current_node->get_next_edges();

            if (next_edges.size() != output_grads.size()) {
                std::string size_edges = std::to_string(next_edges.size());
                std::string size_output_grads = std::to_string(output_grads.size());
                throw std::runtime_error(fmt::format(
                    "{} returned {} gradients, but number of edges is {}",
                    current_node->name(), size_output_grads, size_edges
                ));
            }

            for (size_t output_nr = 0; output_nr < next_edges.size(); output_nr++) {
                const Edge& edge = next_edges[output_nr];
                if (!edge.is_valid()) continue;
                
                TensorList& next_node_inp_grads = grad_map.at(edge.node.get());
                Tensor& target_grad_buffer = next_node_inp_grads[edge.input_nr];

                Tensor& edge_grad = output_grads[output_nr];

                if (!grad.defined()) {
                    throw std::runtime_error(fmt::format(
                        "Computed grad for valid edge is undefined (node={})", current_node->name()
                    ));
                }

                if (target_grad_buffer.defined()) {
                    target_grad_buffer += edge_grad;
                } else {
                    target_grad_buffer = edge_grad;
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
