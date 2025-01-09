
#include "cpu_ops.cpp"

#include <torch/torch.h>
#include <synapx/tensor.hpp>


class Add(Interface) {

public:

    static synapx::Tensor forward(Context& context, const synapx::Tensor t1, const synapx::Tensor t2) {
       
        if t1.device == Device.CPU {
             torch::Tensor out_data = cpu::add_forward(t1.data, t2.data)
        } else {

        }
        
        bool req_grad = true;

        out = Tensor(out_data);

        return out
    }

    static void backward(Context& context) {
        out_grad = context.out_grad
        context.t1_shape
        context.t2_shape

        if t1.device == Device.CPU {
            torch::Tensor a_grad, torch::Tensor b_grad = cpu::add_backward(out_grad, t1.data, t2.data)
        } else {

        }

        context.t1.grad += a_grad
        context.t2.grad += b_grad
    }

};

namespace synapx
{
    Tensor add(const Tensor& t1, const Tensor& t2) {
        // Check input tensors
        Context context = Context();
        Tensor out = Add::forward(context, t1, t2);
       
        if (result.req_grad) {
            out.grad_fn = BackwardFunction(context, Add::backward);
        }

        return out
    }
    
} // namespace synapx
