
#include <synapx/functional.hpp>

#include <vector>

#include <torch/torch.h>

#include <synapx/tensor.hpp>
#include <synapx/device.hpp>
#include <synapx/autograd/functions.hpp>


namespace synapx::F {

    Tensor add(const Tensor& t1, const Tensor& t2) {
        // Check input tensors
        if (!t1.defined() || !t2.defined()) {
            throw std::invalid_argument("Input tensors must be valid");
        }
        
        if (t1.device() != t2.device()) {
            throw std::invalid_argument("Input tensors must be on the same device");
        }

        std::vector<Tensor> inputs{t1, t2};
        
        // Create the add operation
        std::shared_ptr<autograd::Add> add_op;
        if (t1.device() == Device::CPU()) {
            add_op = std::make_shared<autograd::Add>();
        } else {
            throw std::runtime_error("Device not supported: " + t1.device().to_string());
        }

        torch::Tensor result_data = add_op->forward({t1.data(), t2.data()})[0];

        bool req_grad = t1.requires_grad() || t2.requires_grad();
        Tensor result(result_data, req_grad, t1.device());
        
        if (req_grad) {
            result.set_grad_fn(add_op);

            for (size_t i = 0; i < inputs.size(); ++i) {
                Tensor inp = inputs[i];
                autograd::Function::BackwardEdge next_edge {inp.grad_fn(), i, inp};
                add_op->backward_edges.push_back(next_edge);
            }
        }

        return result;
    }

} // namespace synapx
