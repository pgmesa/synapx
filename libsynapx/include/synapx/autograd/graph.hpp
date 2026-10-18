#ifndef AUTOGRAD_GRAPH_HPP
#define AUTOGRAD_GRAPH_HPP

#include <memory>
#include <vector>
#include <utility>
#include <functional>

#include <synapx/tensor.hpp>
#include <synapx/autograd/engine.hpp>


namespace synapx::autograd {

    struct Edge;
    class Node;

    using NodePtr = std::shared_ptr<autograd::Node>;
    using EdgeList = std::vector<Edge>;

    struct SYNAPX_API Edge {
        Edge() : node(nullptr), input_nr(0) {}
        Edge(std::shared_ptr<Node> node, uint32_t input_nr) 
            : node(node), input_nr(input_nr) {};

        bool is_valid() const {
            return node != nullptr;
        }

        std::shared_ptr<Node> node;
        uint32_t input_nr;
    };

    class SYNAPX_API Node {
    public:
        Node() = default;
        virtual ~Node() = default;
        
        // Nodes are neither copyable nor moveable.
        Node(const Node& other) = delete;
        Node(Node&& other) = delete;
        Node& operator=(const Node& other) = delete;
        Node& operator=(Node&& other) = delete;
        
        virtual std::string name() const = 0;

        TensorList operator()(const TensorList& inputs) {
            NoGradGuard guard; // Disable grad
            return apply(inputs);
        };

        void increment_input_count() {
            _num_inputs += 1;
        };

        virtual size_t num_inputs() const { 
            return _num_inputs; 
        };

        void add_next_edge(Edge edge) {
            next_edges.emplace_back(std::move(edge));
        }

        const EdgeList& get_next_edges() const {
            return next_edges;
        }

        size_t num_outputs() const {
            return next_edges.size();
        };

        // Called by the engine with the gradient of output `output_nr` before
        // apply(). Used by Tensor::retain_grad() to populate non-leaf grads.
        using RetainGradHook = std::function<void(const Tensor&)>;

        void add_retain_grad_hook(uint32_t output_nr, RetainGradHook hook) {
            retain_grad_hooks.emplace_back(output_nr, std::move(hook));
        }

        const std::vector<std::pair<uint32_t, RetainGradHook>>& get_retain_grad_hooks() const {
            return retain_grad_hooks;
        }

    private:
        EdgeList next_edges;
        size_t _num_inputs = 0;
        std::vector<std::pair<uint32_t, RetainGradHook>> retain_grad_hooks;

        virtual TensorList apply(const TensorList& inputs) = 0;
    };

} // namespace synapx::autograd

#endif