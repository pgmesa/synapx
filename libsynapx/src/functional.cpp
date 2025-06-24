
#include <synapx/functional.hpp>

#include <vector>

#include <torch/torch.h>

#include <synapx/tensor.hpp>
#include <synapx/autograd/graph.hpp>
#include <synapx/autograd/functions.hpp>


namespace synapx {

    // Initializers
    Tensor empty(torch::IntArrayRef shape, bool requires_grad, torch::TensorOptions options) {
        return Tensor(torch::empty(shape, options), requires_grad);
    };

    Tensor empty_like(Tensor t, bool requires_grad, torch::TensorOptions options) {
        return Tensor(torch::empty_like(t.data(), options), requires_grad);
    };

    Tensor ones(torch::IntArrayRef shape, bool requires_grad, torch::TensorOptions options) {
        return Tensor(torch::ones(shape, options), requires_grad);
    };

    Tensor ones_like(Tensor t, bool requires_grad, torch::TensorOptions options) {
        return Tensor(torch::ones_like(t.data(), options), requires_grad);
    };

    Tensor zeros(torch::IntArrayRef shape, bool requires_grad, torch::TensorOptions options) {
        return Tensor(torch::zeros(shape, options), requires_grad);
    };

    Tensor zeros_like(Tensor t, bool requires_grad, torch::TensorOptions options) {
        return Tensor(torch::zeros_like(t.data(), options), requires_grad);
    };

    Tensor rand(torch::IntArrayRef shape, bool requires_grad, torch::TensorOptions options) {
        return Tensor(torch::rand(shape, options), requires_grad);
    };

    Tensor rand_like(Tensor t, bool requires_grad, torch::TensorOptions options) {
        return Tensor(torch::rand_like(t.data(), options), requires_grad);
    };

    Tensor randn(torch::IntArrayRef shape, bool requires_grad, torch::TensorOptions options) {
        return Tensor(torch::randn(shape, options), requires_grad);
    };

    Tensor randn_like(Tensor t, bool requires_grad, torch::TensorOptions options) {
        return Tensor(torch::randn_like(t.data(), options), requires_grad);
    };

    Tensor full(torch::IntArrayRef shape, double fill_value, bool requires_grad, torch::TensorOptions options) {
        return Tensor(torch::full(shape, fill_value, options), requires_grad);
    };

    Tensor full_like(Tensor t, double fill_value, bool requires_grad, torch::TensorOptions options) {
        return Tensor(torch::full_like(t.data(), fill_value, options), requires_grad);
    };
    
    namespace {

        using TorchList = std::vector<torch::Tensor>;
        using Operation = std::function<TorchList()>;
        using NodeFactory = std::function<autograd::NodePtr(const TensorList&)>;

        TensorList apply_operation(TensorList inputs, Operation operation, NodeFactory node_factory) {
            if (inputs.empty())
                throw std::invalid_argument("At least one input is required");

            bool any_grad = false;
            for (const Tensor& t : inputs) {
                if (!t.defined())
                    throw std::invalid_argument("Input tensors must be valid");
                
                any_grad = autograd::is_grad_enabled() && (any_grad || t.requires_grad());
            }
            
            std::vector<torch::Tensor> outputs_data = operation();
            
            TensorList outputs;
            for (size_t i = 0; i < outputs_data.size(); ++i) {
                outputs.emplace_back(outputs_data[i], any_grad);
                outputs.back().set_output_nr(static_cast<uint32_t>(i));
            }

            if (any_grad) {
                // Create node
                autograd::NodePtr node = node_factory(outputs);

                // Link node to graph
                for (const Tensor& input : inputs) {
                    autograd::Edge edge;
                    if (input.grad_fn()) {
                        edge = {input.grad_fn(), input.output_nr()};
                    } 
                    else if (input.is_leaf() && input.requires_grad()) {
                        autograd::NodePtr accum_node = std::make_shared<autograd::AccumulateGrad>(input);
                        accum_node->increment_input_count();
                        edge = {accum_node, 0};
                    }
                    node->add_next_edge(edge);
                }

                for (Tensor& output : outputs) {
                    output.set_grad_fn(node);
                    node->increment_input_count();
                }
            }
            
            return outputs;
        }
    }

    // Basic Functions
    Tensor add(const Tensor& t1, const Tensor& t2) {
        TensorList inputs {t1, t2};

        Operation operation = [&t1, &t2]() -> TorchList {
            torch::Tensor output = torch::add(t1.data(), t2.data());
            return { output };
        };

        NodeFactory node_factory = [&t1, &t2](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::AddBackward0>(t1, t2);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor add(const Tensor& t1, double t2) {
        return add(t1, Tensor(torch::tensor(t2, t1.options()), false));
    }


    Tensor sub(const Tensor& t1, const Tensor& t2) {
        TensorList inputs {t1, t2};

        Operation operation = [&t1, &t2]() -> TorchList {
            torch::Tensor output = torch::sub(t1.data(), t2.data());
            return { output };
        };

        NodeFactory node_factory = [&t1, &t2](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::SubBackward0>(t1, t2);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor sub(const Tensor& t1, double t2) {
        return sub(t1, Tensor(torch::tensor(t2, t1.options()), false));
    }


    Tensor mul(const Tensor& t1, const Tensor& t2) {
        TensorList inputs {t1, t2};

        Operation operation = [&t1, &t2]() -> TorchList {
            torch::Tensor output = torch::mul(t1.data(), t2.data());
            return { output };
        };

        NodeFactory node_factory = [&t1, &t2](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::MulBackward0>(t1, t2);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor mul(const Tensor& t1, double t2) {
        return mul(t1, Tensor(torch::tensor(t2, t1.options()), false));
    }


    Tensor div(const Tensor& t1, const Tensor& t2) {
        TensorList inputs {t1, t2};

        Operation operation = [&t1, &t2]() -> TorchList {
            torch::Tensor output = torch::div(t1.data(), t2.data());
            return { output };
        };

        NodeFactory node_factory = [&t1, &t2](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::DivBackward0>(t1, t2);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor div(const Tensor& t1, double t2) {
        return div(t1, Tensor(torch::tensor(t2, t1.options()), false));
    }
    

    Tensor matmul(const Tensor& t1, const Tensor& t2) {
        TensorList inputs {t1, t2};

        Operation operation = [&t1, &t2]() -> TorchList {
            torch::Tensor output = torch::matmul(t1.data(), t2.data());
            return { output };
        };

        NodeFactory node_factory;
        if (t1.dim() == 1 && t2.dim() == 1) {
            node_factory = [&t1, &t2](const TensorList& outputs) -> autograd::NodePtr {
                return std::make_shared<autograd::NotImplementedBackward>();
            };
        } else {
            node_factory = [&t1, &t2](const TensorList& outputs) -> autograd::NodePtr {
                return std::make_shared<autograd::MatmulBackward0>(t1, t2);
            };
        }

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }


    Tensor pow(const Tensor& base, const Tensor& exp) {
        TensorList inputs {base, exp};

        Operation operation = [&base, &exp]() -> TorchList {
            torch::Tensor output = torch::pow(base.data(), exp.data());
            return { output };
        };

        NodeFactory node_factory = [&base, &exp](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::PowBackward0>(base, exp, outputs[0]);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor pow(const Tensor& base, double exp) {
        return pow(base, Tensor(torch::tensor(exp, base.options()), false));
    }


    Tensor neg(const Tensor& t) {
        return mul(t, Tensor(torch::tensor(-1.0, t.options()), false));
    }


    // Reverse Functions
    Tensor rsub(const Tensor& t1, const Tensor& t2) {
        return sub(t2, t1);
    };

    Tensor rsub(const Tensor& t1, double t2) {
        return sub(Tensor(torch::tensor(t2, t1.options()), false), t1);
    };

    Tensor rpow(const Tensor& exp, const Tensor& base) {
        return pow(base, exp);
    };

    Tensor rpow(const Tensor& exp, double base) {
        return rpow(exp, Tensor(torch::tensor(base, exp.options()), false));
    };

    Tensor rdiv(const Tensor& t1, const Tensor& t2) {
        return div(t2, t1);
    };

    Tensor rdiv(const Tensor& t1, double t2) {
        return div(Tensor(torch::tensor(t2, t1.options()), false), t1);
    };

    Tensor rmatmul(const Tensor& t1, const Tensor& t2) {
        return matmul(t2, t1);
    };


    // Other functions
    Tensor copy_to(const Tensor& t, torch::Device device) {
        TensorList inputs {t};
        
        Operation operation = [&t, device]() -> TorchList {
            return { t.data().to(device) };
        };

        NodeFactory node_factory = [&t](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::ToCopyBackward0>(t.device());
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor copy_to(const Tensor& t, torch::Dtype dtype) {
        TensorList inputs {t};

        Operation operation = [&t, dtype]() -> TorchList {
            return { t.data().to(dtype) };
        };

        NodeFactory node_factory = [&t](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::ToCopyBackward0>(t.dtype());
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }


    Tensor clone(const Tensor& t) {
        TensorList inputs {t};

        Operation operation = [&t]() -> TorchList {
            return { torch::clone(t.data()) };
        };

        NodeFactory node_factory = [](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::CloneBackward0>();
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor addmm(const Tensor& inp, const Tensor& mat1, const Tensor& mat2) {
        TensorList inputs {inp, mat1, mat2};

        Operation operation = [&inp, &mat1, &mat2]() -> TorchList {
            torch::Tensor output = torch::addmm(inp.data(), mat1.data(), mat2.data());
            return { output };
        };

        NodeFactory node_factory = [&inp, &mat1, &mat2](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::AddmmBackward0>(inp, mat1, mat2);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor exp(const Tensor& t) {
        TensorList inputs {t};

        Operation operation = [&t]() -> TorchList {
            return { torch::exp(t.data()) };
        };

        NodeFactory node_factory = [&t](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::ExpBackward0>(outputs[0]);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor log(const Tensor& t) {
        TensorList inputs {t};

        Operation operation = [&t]() -> TorchList {
            return { torch::log(t.data()) };
        };

        NodeFactory node_factory = [&t](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::LogBackward0>(t);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor sqrt(const Tensor& t) {
        TensorList inputs {t};

        Operation operation = [&t]() -> TorchList {
            return { torch::sqrt(t.data()) };
        };

        NodeFactory node_factory = [&t](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::SqrtBackward0>(outputs[0]);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor sum(const Tensor& t, torch::IntArrayRef dim, bool keepdim) {
        TensorList inputs {t};

        Operation operation = [&t, dim, keepdim]() -> TorchList {
            return { torch::sum(t.data(), dim, keepdim) };
        };

        NodeFactory node_factory = [&t, dim, keepdim](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::SumBackward0>(t, dim, keepdim);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor mean(const Tensor& t, torch::IntArrayRef dim, bool keepdim) {
        TensorList inputs {t};

        Operation operation = [&t, dim, keepdim]() -> TorchList {
            return { torch::mean(t.data(), dim, keepdim) };
        };

        NodeFactory node_factory = [&t, dim, keepdim](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::MeanBackward0>(t, dim, keepdim);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor max(const Tensor& t) {
        TensorList inputs {t};

        Operation operation = [&t]() -> TorchList {
            return { torch::max(t.data()) };
        };

        NodeFactory node_factory = [&t](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::MaxBackward1>(t, outputs[0]);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    std::tuple<Tensor, Tensor> max(const Tensor& t, int64_t dim, bool keepdim) {
        TensorList inputs {t};

        Tensor max_indices;
        Operation operation = [&t, dim, keepdim, &max_indices]() -> TorchList {
            auto [output, indices] = torch::max(t.data(), dim, keepdim);
            max_indices = indices;
            return { output };
        };

        NodeFactory node_factory = [&t, dim, keepdim, &max_indices](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::MaxBackward0>(t, dim, keepdim, max_indices);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];
        
        return {output, max_indices};
    }

    Tensor min(const Tensor& t) {
        TensorList inputs {t};

        Operation operation = [&t]() -> TorchList {
            return { torch::min(t.data()) };
        };

        NodeFactory node_factory = [&t](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::MinBackward1>(t, outputs[0]);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    std::tuple<Tensor, Tensor> min(const Tensor& t, int64_t dim, bool keepdim) {
        TensorList inputs {t};

        Tensor min_indices;
        Operation operation = [&t, dim, keepdim, &min_indices]() -> TorchList {
            auto [output, indices] = torch::min(t.data(), dim, keepdim);
            min_indices = indices;
            return { output };
        };

        NodeFactory node_factory = [&t, dim, keepdim, &min_indices](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::MinBackward0>(t, dim, keepdim, min_indices);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];
        
        return {output, min_indices};
    }

    Tensor squeeze(const Tensor& t, torch::IntArrayRef dim) {
        TensorList inputs {t};

        Operation operation = [&t, dim]() -> TorchList {
            torch::Tensor result = dim.empty()? torch::squeeze(t.data()) : torch::squeeze(t.data(), dim); 
            return { result };
        };

        NodeFactory node_factory = [&t, dim](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::SqueezeBackward0>(t, dim);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor unsqueeze(const Tensor& t, int64_t dim) {
        TensorList inputs {t};

        Operation operation = [&t, dim]() -> TorchList {
            return { torch::unsqueeze(t.data(), dim) };
        };

        NodeFactory node_factory = [dim](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::UnsqueezeBackward0>(dim);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor reshape(const Tensor& t, torch::IntArrayRef shape) {
        TensorList inputs {t};

        Operation operation = [&t, shape]() -> TorchList {
            return { torch::reshape(t.data(), shape) };
        };

        NodeFactory node_factory = [&t](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::ReshapeBackward0>(t);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor broadcast_to(const Tensor& t, torch::IntArrayRef shape) {
        TensorList inputs {t};

        Operation operation = [&t, shape]() -> TorchList {
            return { torch::broadcast_to(t.data(), shape) };
        };

        NodeFactory node_factory = [&t, shape](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::NotImplementedBackward>();
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor transpose(const Tensor& t, int64_t dim0, int64_t dim1) {
        TensorList inputs {t};

        Operation operation = [&t, dim0, dim1]() -> TorchList {
            return { torch::transpose(t.data(), dim0, dim1) };
        };

        NodeFactory node_factory = [dim0, dim1](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::TransposeBackward0>(dim0, dim1);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor swapdims(const Tensor& t, int64_t dim0, int64_t dim1) {
        return transpose(t, dim0, dim1);
    }

    Tensor movedim(const Tensor& t, int64_t src, int64_t dest) {
        TensorList inputs {t};

        Operation operation = [&t, src, dest]() -> TorchList {
            return { torch::movedim(t.data(), src, dest) };
        };

        NodeFactory node_factory = [src, dest](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::MovedimBackward0>(src, dest);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }
    
    Tensor slice(const Tensor& t, const TensorIndices& indices) {
        TensorList inputs {t};

        Operation operation = [&t, &indices]() -> TorchList {
            return { t.data().index(indices) };
        };

        NodeFactory node_factory = [&t, &indices](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::SliceBackward0>(t, indices);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor select(const Tensor& t, int64_t dim, int64_t index) {
        TensorList inputs {t};

        Operation operation = [&t, dim, index]() -> TorchList {
            return { t.data().select(dim, index) };
        };

        NodeFactory node_factory = [](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::NotImplementedBackward>();
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor concat(const TensorList& tensors, int64_t dim) {
        const TensorList& inputs = tensors;

        TorchList torch_tensors;
        torch_tensors.reserve(tensors.size());
        for (const Tensor& t : tensors)
            torch_tensors.push_back(t.data());

        Operation operation = [&torch_tensors, dim]() -> TorchList {
            return { torch::concat(torch_tensors, dim) };
        };

        NodeFactory node_factory = [&inputs, dim](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::ConcatBackward0>(inputs, dim);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor stack(const TensorList& tensors, int64_t dim) {
        const TensorList& inputs = tensors;

        TorchList torch_tensors;
        torch_tensors.reserve(tensors.size());
        for (const Tensor& t : tensors)
            torch_tensors.push_back(t.data());

        Operation operation = [&torch_tensors, dim]() -> TorchList {
            return { torch::stack(torch_tensors, dim) };
        };

        NodeFactory node_factory = [dim](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::StackBackward0>(dim);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    TensorList unbind(const Tensor& t, int64_t dim) {
        const TensorList inputs {t};

        Operation operation = [&t, dim]() -> TorchList {
            return torch::unbind(t.data(), dim);
        };

        NodeFactory node_factory = [&t, dim](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::UnbindBackward0>(t, dim);
        };

        TensorList outputs = apply_operation(inputs, operation, node_factory);

        return outputs;
    }

    TensorList split(const Tensor& t, torch::IntArrayRef split_size, int64_t dim) {
        const TensorList inputs {t};

        Operation operation = [&t, split_size, dim]() -> TorchList {
            return torch::split(t.data(), split_size, dim);
        };

        NodeFactory node_factory = [](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::NotImplementedBackward>();
        };

        TensorList outputs = apply_operation(inputs, operation, node_factory);

        return outputs;
    }


    Tensor diag(const Tensor& t, int64_t diagonal) {
        TensorList inputs {t};

        Operation operation = [&t, diagonal]() -> TorchList {
            return { torch::diag(t.data(), diagonal) };
        };

        NodeFactory node_factory = [](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::NotImplementedBackward>();
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }


    Tensor outer(const Tensor& input, const Tensor& vec2) {
        TensorList inputs {input, vec2};

        Operation operation = [&input, &vec2]() -> TorchList {
            return { torch::outer(input.data(), vec2.data()) };
        };

        NodeFactory node_factory = [](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::NotImplementedBackward>();
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor where(const Tensor& condition, const Tensor& input, const Tensor& other) {
        TensorList inputs {condition, input, other};

        Operation operation = [&condition, &input, &other]() -> TorchList {
            return { torch::where(condition.data(), input.data(), other.data()) };
        };

        NodeFactory node_factory = [](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::NotImplementedBackward>();
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor where(const Tensor& condition, const Tensor& input, double other) {
        return where(condition, input, Tensor(torch::tensor(other)));
    }


    // Activations
    Tensor relu(const Tensor& t) {
        const TensorList inputs {t};

        Operation operation = [&t]() -> TorchList {
            return { t.data().relu() };
        };

        NodeFactory node_factory = [&t](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::ReLUBackward0>(outputs[0]);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor sigmoid(const Tensor& t) {
        const TensorList inputs {t};

        Operation operation = [&t]() -> TorchList {
            return { t.data().sigmoid() };
        };

        NodeFactory node_factory = [](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::SigmoidBackward0>(outputs[0]);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    SYNAPX_API Tensor softmax(const Tensor& t, int64_t dim) {
        const TensorList inputs {t};

        Operation operation = [&t, dim]() -> TorchList {
            return { t.data().softmax(dim) };
        };

        NodeFactory node_factory = [dim](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::SoftmaxBackward0>(outputs[0], dim);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    SYNAPX_API Tensor log_softmax(const Tensor& t, int64_t dim) {
        const TensorList inputs {t};

        Operation operation = [&t, dim]() -> TorchList {
            return { t.data().log_softmax(dim) };
        };

        NodeFactory node_factory = [dim](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::LogSoftmaxBackward0>(outputs[0], dim);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    namespace {

        torch::Tensor apply_reduction(torch::Tensor tensor, Reduction reduction) {
            if (reduction == Reduction::Mean) return tensor.mean();
            else if (reduction == Reduction::Sum) return tensor.sum();
            return tensor;
        }

    }

    // Losses
    Tensor mse_loss(const Tensor& input, const Tensor& target, Reduction reduction) {
        TensorList inputs {input, target};

        Tensor diff;
        Operation operation = [&input, &target, &diff, reduction]() -> TorchList {
            torch::Tensor diff_data = input.data() - target.data();
            diff = Tensor(diff_data);
            return { apply_reduction(diff_data.pow(2), reduction) };
        };

        NodeFactory node_factory = [&input, &target, &diff, reduction](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::MSELossBackward0>(input, target, diff, reduction);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor nll_loss(const Tensor& input, const Tensor& target, Reduction reduction) {
        TensorList inputs {input, target};

        Operation operation = [&input, &target, reduction]() -> TorchList {
            return { torch::nll_loss(input.data(), target.data(), std::nullopt, reduction) };
        };

        NodeFactory node_factory = [&input, &target, reduction](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::NLLLossBackward0>(input, target, reduction);
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }


    // Layer operations
    Tensor linear(const Tensor& inp, const Tensor& weight, std::optional<Tensor> bias) {
        return bias.has_value()? addmm(bias.value(), inp, weight.t()) : matmul(inp, weight.t());
    }

    Tensor flatten(const Tensor& inp, int64_t start_dim, int64_t end_dim) {
        int64_t rank = static_cast<int64_t>(inp.dim());
        int64_t start = start_dim >= 0 ? start_dim : start_dim + rank;
        int64_t end = end_dim >= 0 ? end_dim : end_dim + rank;

        if (start > end || start < 0 || end >= rank)
            throw std::runtime_error("flatten() has invalid args: start_dim or end_dim is out of bounds");

        IntArray new_shape;
        int64_t flattened_dim = 1;

        for (int64_t i = 0; i < rank; ++i) {
            if (i < start || i > end) {
                new_shape.push_back(inp.size(i));
            } else {
                flattened_dim *= inp.size(i);
            }
        }

        new_shape.insert(new_shape.begin() + start, flattened_dim);

        return reshape(inp, new_shape);
    }

    Tensor dropout(const Tensor& t, double p, bool train) {
        if (!train || p == 0.0) return t;

        if (p < 0.0 || p >= 1.0)
            throw std::runtime_error("dropout(): p must be in the range [0.0, 1.0)");

        // Create a mask with the same shape as the input tensor
        double keep_prob = 1.0 - p;
        Tensor mask = (synapx::rand_like(t) < keep_prob).to(t.dtype());  // binary mask
        mask = mask / keep_prob; // Scale the mask to preserve the expected value

        return t * mask;
    }


} // namespace synapx
