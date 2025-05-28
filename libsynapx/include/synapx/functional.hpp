#ifndef FUNCTIONAL_HPP
#define FUNCTIONAL_HPP

#include <synapx/core.hpp>
#include <synapx/tensor.hpp>

namespace synapx::F {

    SYNAPX_API Tensor add(const Tensor& t1, const Tensor& t2);
    SYNAPX_API Tensor mul(const Tensor& t1, const Tensor& t2);
    SYNAPX_API Tensor matmul(const Tensor& t1, const Tensor& t2);

    namespace detail {

        template<class Factory>
        std::vector<Tensor> dispatch_op(const std::vector<Tensor>& inputs, Factory make_fn) {
            if (inputs.empty())
                throw std::invalid_argument("dispatch_op: at least one input required");

            // 1) all inputs must be defined and on the same device
            auto dev = inputs[0].device();
            bool any_grad = false;
            for (auto& t : inputs) {
                if (!t.defined())
                    throw std::invalid_argument("Input tensors must be valid");
                    
                if (t.device() != dev)
                    throw std::invalid_argument("All inputs must be on the same device");
                
                any_grad = any_grad || t.requires_grad();
            }

            // 2) raw torch::Tensor inputs
            std::vector<torch::Tensor> raw_in;
            raw_in.reserve(inputs.size());
            for (auto& t : inputs)
                raw_in.push_back(t.data());

            // 3) pick/construct the right Function
            auto fn = make_fn(dev);
            if (!fn)
                throw std::runtime_error("dispatch_op: factory returned nullptr");

            // 4) forward-compute
            auto raw_out = fn->forward(raw_in);

            // 5) wrap outputs & hook up grad_fn/backward_edges
            std::vector<Tensor> outputs;
            outputs.reserve(raw_out.size());
            for (auto& ro : raw_out)
                outputs.emplace_back(ro, any_grad, dev);

            if (any_grad) {
                // set the same grad_fn on each output
                for (auto& out : outputs)
                    out.set_grad_fn(fn);

                // record one backward‐edge per input slot
                for (size_t i = 0; i < inputs.size(); ++i) {
                    fn->backward_edges.push_back({inputs[i].grad_fn(), i, inputs[i]});
                }
            }

            return outputs;
        }

    } // Detail

}

#endif // FUNCTIONAL_HPP