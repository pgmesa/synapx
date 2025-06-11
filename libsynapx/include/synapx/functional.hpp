#ifndef FUNCTIONAL_HPP
#define FUNCTIONAL_HPP

#include <optional>
#include <tuple>
#include <memory>

#include <synapx/core.hpp>
#include <synapx/tensor.hpp>
#include <synapx/autograd/engine.hpp>

namespace synapx {

    SYNAPX_API Tensor add(const Tensor& t1, const Tensor& t2);
    SYNAPX_API Tensor add(const Tensor& t1, double t2);
    SYNAPX_API Tensor sub(const Tensor& t1, const Tensor& t2);
    SYNAPX_API Tensor sub(const Tensor& t1, double t2);
    SYNAPX_API Tensor mul(const Tensor& t1, const Tensor& t2);
    SYNAPX_API Tensor mul(const Tensor& t1, double t2);
    SYNAPX_API Tensor pow(const Tensor& t1, const Tensor& exp);
    SYNAPX_API Tensor pow(const Tensor& t1, double exp);
    SYNAPX_API Tensor div(const Tensor& t1, const Tensor& t2);
    SYNAPX_API Tensor div(const Tensor& t1, double t2);
    SYNAPX_API Tensor matmul(const Tensor& t1, const Tensor& t2);
    SYNAPX_API Tensor neg(const Tensor& t1);

    SYNAPX_API Tensor rsub(const Tensor& t1, const Tensor& t2);
    SYNAPX_API Tensor rsub(const Tensor& t1, double t2);
    SYNAPX_API Tensor rpow(const Tensor& t1, const Tensor& exp);
    SYNAPX_API Tensor rpow(const Tensor& t1, double exp);
    SYNAPX_API Tensor rdiv(const Tensor& t1, const Tensor& t2);
    SYNAPX_API Tensor rdiv(const Tensor& t1, double t2);
    SYNAPX_API Tensor rmatmul(const Tensor& t1, const Tensor& t2);

    SYNAPX_API Tensor addmm(const Tensor& inp, const Tensor& mat1, const Tensor& mat2);
    SYNAPX_API Tensor clone(const Tensor& t);
    SYNAPX_API Tensor exp(const Tensor& t);
    SYNAPX_API Tensor log(const Tensor& t);
    SYNAPX_API Tensor sqrt(const Tensor& t);
    SYNAPX_API Tensor sum(const Tensor& t, const torch::IntArrayRef& dim = {}, bool keepdim = false);
    SYNAPX_API Tensor mean(const Tensor& t, const torch::IntArrayRef& dim = {}, bool keepdim = false);
    SYNAPX_API Tensor max(const Tensor& t);
    SYNAPX_API std::tuple<Tensor, Tensor> max(const Tensor& t, int64_t dim, bool keepdim = false);
    SYNAPX_API Tensor min(const Tensor& t);
    SYNAPX_API std::tuple<Tensor, Tensor> min(const Tensor& t, int64_t dim, bool keepdim = false);
    SYNAPX_API Tensor squeeze(const Tensor& t, const torch::IntArrayRef& dim = {});
    SYNAPX_API Tensor unsqueeze(const Tensor& t, int64_t dim);
    SYNAPX_API Tensor reshape(const Tensor& t, const torch::IntArrayRef& shape);
    SYNAPX_API Tensor transpose(const Tensor& t, int64_t dim0, int64_t dim1);
    SYNAPX_API Tensor movedim(const Tensor& t, int64_t src, int64_t dest);
    SYNAPX_API Tensor slice(const Tensor& t, const std::vector<torch::indexing::TensorIndex>& idx);
    SYNAPX_API Tensor concat(const std::vector<Tensor>& tensors, int64_t dim = 0);
    SYNAPX_API Tensor stack(const std::vector<Tensor>& tensors, int64_t dim = 0);
    SYNAPX_API std::vector<Tensor> unbind(const Tensor& t, int64_t dim = 0);


    namespace detail {

        struct DispatcherOutput {
            std::vector<Tensor> outputs;
            std::shared_ptr<autograd::Function> fn;
        };

        template<class Factory>
        DispatcherOutput dispatch_op(const std::vector<Tensor>& inputs, Factory make_fn) {
            if (inputs.empty())
                throw std::invalid_argument("dispatch_op: at least one input required");

            // 1) all inputs must be defined and on the same device
            std::vector<torch::Tensor> raw_in; 
            raw_in.reserve(inputs.size());
            std::vector<bool> grad_flags; 
            grad_flags.reserve(inputs.size());
            
            Device dev = inputs[0].device();
            bool any_grad = false;
            for (const Tensor& t : inputs) {
                if (!t.defined())
                    throw std::invalid_argument("Input tensors must be valid");
                    
                if (t.device() != dev)
                    throw std::invalid_argument("All inputs must be on the same device");
                
                any_grad = any_grad || t.requires_grad();
                grad_flags.push_back(t.requires_grad());
                raw_in.push_back(t.data());
            }

            // 3) pick/construct the right Function
            std::shared_ptr<autograd::Function> fn = make_fn(dev);
            if (!fn)
                throw std::runtime_error("Internal error in 'dispatch_op': factory returned nullptr");

            // 4) forward-compute
            fn->requires_grad_flags = grad_flags;
            std::vector<torch::Tensor> raw_out = fn->forward(raw_in);

            // 5) wrap outputs & hook up grad_fn/backward_edges
            std::vector<Tensor> outputs;
            outputs.reserve(raw_out.size());
            for (torch::Tensor& ro : raw_out)
                outputs.emplace_back(ro, any_grad, dev);

            if (any_grad) {
                // record one backward‐edge per input slot
                std::vector<autograd::BackwardEdge> backward_edges;
                for (size_t i = 0; i < inputs.size(); ++i) {
                    Tensor input = inputs[i];
                    if (input.requires_grad())
                        backward_edges.push_back({input.grad_fn(), i, input});
                }

                // set the grad_fn on each output
                for (int out_idx = 0; out_idx < outputs.size(); out_idx++) {
                    auto grad_fn = std::make_shared<autograd::BackwardNode>(fn, out_idx, backward_edges);
                    synapx::Tensor& out = outputs[out_idx];
                    out.set_grad_fn(grad_fn);
                }
            }
            
            return {outputs, fn};
        }

    } // Detail

}

#endif // FUNCTIONAL_HPP