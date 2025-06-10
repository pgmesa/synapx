
#include <synapx/functional.hpp>

#include <vector>

#include <torch/torch.h>

#include <synapx/tensor.hpp>
#include <synapx/device.hpp>
#include <synapx/autograd/cpu/ops.hpp>


namespace synapx {

    Tensor add(const Tensor& t1, const Tensor& t2) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            {t1, t2}, 
            [](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Add>();
                throw std::runtime_error("Add: unsupported device");
            }
        );
        return dout.outputs[0];
    }

    Tensor add(const Tensor& t1, double t2) {
        return add(t1, Tensor(torch::tensor(t2, t1.data().options()), false, t1.device()));
    }

    Tensor sub(const Tensor& t1, const Tensor& t2) {
        return add(t1, t2 * -1);
    }

    Tensor sub(const Tensor& t1, double t2) {
        return sub(t1, Tensor(torch::tensor(t2, t1.data().options()), false, t1.device()));
    }

    Tensor mul(const Tensor& t1, const Tensor& t2) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            {t1, t2}, 
            [](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Mul>();
                throw std::runtime_error("Mul: unsupported device");
            }
        );
        return dout.outputs[0];
    }

    Tensor mul(const Tensor& t1, double t2) {
        return mul(t1, Tensor(torch::tensor(t2, t1.data().options()), false, t1.device()));
    }

    Tensor div(const Tensor& t1, const Tensor& t2) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            {t1, t2}, 
            [](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Div>();
                throw std::runtime_error("Div: unsupported device");
            }
        );
        return dout.outputs[0];
    }

    Tensor div(const Tensor& t1, double t2) {
        return div(t1, Tensor(torch::tensor(t2, t1.data().options()), false, t1.device()));
    }

    Tensor matmul(const Tensor& t1, const Tensor& t2) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            {t1, t2}, 
            [](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Matmul>();
                throw std::runtime_error("Matmul: unsupported device");
            }
        );
        return dout.outputs[0];
    }

    Tensor pow(const Tensor& t1, const Tensor& exp) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            {t1, exp}, 
            [](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Pow>();
                throw std::runtime_error("Pow: unsupported device");
            }
        );
        return dout.outputs[0];
    }

    Tensor pow(const Tensor& t1, double exp) {
        return pow(t1, Tensor(torch::tensor(exp, t1.data().options()), false, t1.device()));
    }

    Tensor neg(const Tensor& t1) {
        return mul(t1, Tensor(torch::tensor(-1.0, t1.data().options()), false, t1.device()));
    }


    // Reverse Functions
    Tensor rsub(const Tensor& t1, const Tensor& t2) {
        return sub(t2, t1);
    };

    Tensor rsub(const Tensor& t1, double t2) {
        return sub(Tensor(torch::tensor(t2, t1.data().options()), false, t1.device()), t1);
    };

    Tensor rpow(const Tensor& exp, const Tensor& base) {
        return pow(base, exp);
    };

    Tensor rpow(const Tensor& exp, double base) {
        return rpow(exp, Tensor(torch::tensor(base, exp.data().options()), false, exp.device()));
    };

    Tensor rdiv(const Tensor& t1, const Tensor& t2) {
        return div(t2, t1);
    };

    Tensor rdiv(const Tensor& t1, double t2) {
        return div(Tensor(torch::tensor(t2, t1.data().options()), false, t1.device()), t1);
    };

    Tensor rmatmul(const Tensor& t1, const Tensor& t2) {
        return matmul(t2, t1);
    };


    // Other functions
    Tensor addmm(const Tensor& inp, const Tensor& mat1, const Tensor& mat2) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            {inp, mat1, mat2}, 
            [](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Addmm>();
                throw std::runtime_error("Addmm: unsupported device");
            }
        );
        return dout.outputs[0];
    }

    Tensor clone(const Tensor& t) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            {t}, 
            [](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Clone>();
                throw std::runtime_error("Clone: unsupported device");
            }
        );
        return dout.outputs[0];
    }

    Tensor exp(const Tensor& t) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            {t}, 
            [](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Exp>();
                throw std::runtime_error("Exp: unsupported device");
            }
        );
        return dout.outputs[0];
    }

    Tensor log(const Tensor& t) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            {t}, 
            [](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Log>();
                throw std::runtime_error("Log: unsupported device");
            }
        );
        return dout.outputs[0];
    }

    Tensor sqrt(const Tensor& t) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            {t}, 
            [](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Sqrt>();
                throw std::runtime_error("Sqrt: unsupported device");
            }
        );
        return dout.outputs[0];
    }

    Tensor sum(const Tensor& t, const torch::IntArrayRef& dim, bool keepdim) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            {t}, 
            [dim, keepdim](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Sum>(dim, keepdim);
                throw std::runtime_error("Sum: unsupported device");
            }
        );
        return dout.outputs[0];
    }

    Tensor mean(const Tensor& t, const torch::IntArrayRef& dim, bool keepdim) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            {t}, 
            [dim, keepdim](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Mean>(dim, keepdim);
                throw std::runtime_error("Mean: unsupported device");
            }
        );
        return dout.outputs[0];
    }

    Tensor max(const Tensor& t) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            {t}, 
            [](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Max>();
                throw std::runtime_error("Max: unsupported device");
            }
        );
        return dout.outputs[0];
    }

    std::tuple<Tensor, Tensor> max(const Tensor& t, int64_t dim, bool keepdim) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            {t}, 
            [dim, keepdim](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Max>(dim, keepdim);
                throw std::runtime_error("Max: unsupported device");
            }
        );

        // This is not the best way to do it, but sufficies for now. If more devices
        // are added this will be done in a different way. 
        torch::Tensor max_indices;
        if (t.device().type() == Device::Type::CPU) {
            auto max_fn = std::dynamic_pointer_cast<autograd::cpu::Max>(dout.fn);

            if (!max_fn) {
                throw std::runtime_error("Failed to cast to autograd::cpu::Max");
            }

            max_indices = max_fn->max_indices;
        }
        
        return {dout.outputs[0], max_indices};
    }

    Tensor min(const Tensor& t) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            {t}, 
            [](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Min>();
                throw std::runtime_error("Min: unsupported device");
            }
        );
        return dout.outputs[0];
    }

    std::tuple<Tensor, Tensor> min(const Tensor& t, int64_t dim, bool keepdim) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            {t}, 
            [dim, keepdim](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Min>(dim, keepdim);
                throw std::runtime_error("Min: unsupported device");
            }
        );

        // This is not the best way to do it, but sufficies for now. If more devices
        // are added this will be done in a different way. 
        torch::Tensor min_indices;
        if (t.device().type() == Device::Type::CPU) {
            auto min_fn = std::dynamic_pointer_cast<autograd::cpu::Min>(dout.fn);

            if (!min_fn) {
                throw std::runtime_error("Failed to cast to autograd::cpu::Min");
            }

            min_indices = min_fn->min_indices;
        }
        
        return {dout.outputs[0], min_indices};
    }

    Tensor squeeze(const Tensor& t, const torch::IntArrayRef& dim) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            {t}, 
            [dim](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Squeeze>(dim);
                throw std::runtime_error("Squeeze: unsupported device");
            }
        );
        return dout.outputs[0];
    }

    Tensor unsqueeze(const Tensor& t, int64_t dim) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            {t},
            [dim](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Unsqueeze>(dim);
                throw std::runtime_error("Unsqueeze: unsupported device");
            }
        );
        return dout.outputs[0];
    }

    Tensor reshape(const Tensor& t, const torch::IntArrayRef& shape) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            {t},
            [shape](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Reshape>(shape);
                throw std::runtime_error("Reshape: unsupported device");
            }
        );
        return dout.outputs[0];
    }

    Tensor transpose(const Tensor& t, int64_t dim0, int64_t dim1) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            {t},
            [dim0, dim1](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Transpose>(dim0, dim1);
                throw std::runtime_error("Transpose: unsupported device");
            }
        );
        return dout.outputs[0];
    }

    Tensor movedim(const Tensor& t, int64_t src, int64_t dest) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            {t},
            [src, dest](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Movedim>(src, dest);
                throw std::runtime_error("Movedim: unsupported device");
            }
        );
        return dout.outputs[0];
    }
    
    Tensor slice(const Tensor& t, const std::vector<torch::indexing::TensorIndex>& idx) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            {t},
            [idx](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Slice>(idx);
                throw std::runtime_error("Slice: unsupported device");
            }
        );
        return dout.outputs[0];
    }

    Tensor concat(const std::vector<Tensor>& tensors, int64_t dim) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            tensors,
            [dim](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Concat>(dim);
                throw std::runtime_error("Concat: unsupported device");
            }
        );
        return dout.outputs[0];
    }

    Tensor stack(const std::vector<Tensor>& tensors, int64_t dim) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            tensors,
            [dim](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Stack>(dim);
                throw std::runtime_error("Stack: unsupported device");
            }
        );
        return dout.outputs[0];
    }

    std::vector<Tensor> unbind(const Tensor& t, int64_t dim) {
        detail::DispatcherOutput dout = detail::dispatch_op(
            {t},
            [dim](Device dev) -> std::shared_ptr<autograd::Function> {
                if (dev.type() == Device::Type::CPU)  return std::make_shared<autograd::cpu::Unbind>(dim);
                throw std::runtime_error("Unbind: unsupported device");
            }
        );
        return dout.outputs;
    }

} // namespace synapx
