
#include <synapx/functional.hpp>

#include <vector>

#include <torch/torch.h>

#include <synapx/tensor.hpp>
#include <synapx/autograd/graph.hpp>
#include <synapx/autograd/functions.hpp>


namespace synapx {

    // Initializers
    Tensor empty(torch::IntArrayRef shape, torch::TensorOptions options, bool requires_grad) {
        return Tensor(torch::empty(shape, options), requires_grad);
    };

    Tensor empty_like(Tensor t, torch::TensorOptions options, bool requires_grad) {
        return Tensor(torch::empty_like(t.data(), options), requires_grad);
    };

    Tensor ones(torch::IntArrayRef shape, torch::TensorOptions options, bool requires_grad) {
        return Tensor(torch::ones(shape, options), requires_grad);
    };

    Tensor ones_like(Tensor t, torch::TensorOptions options, bool requires_grad) {
        return Tensor(torch::ones_like(t.data(), options), requires_grad);
    };

    Tensor zeros(torch::IntArrayRef shape, torch::TensorOptions options, bool requires_grad) {
        return Tensor(torch::zeros(shape, options), requires_grad);
    };

    Tensor zeros_like(Tensor t, torch::TensorOptions options, bool requires_grad) {
        return Tensor(torch::zeros_like(t.data(), options), requires_grad);
    };

    Tensor rand(torch::IntArrayRef shape, torch::TensorOptions options, bool requires_grad) {
        return Tensor(torch::rand(shape, options), requires_grad);
    };

    Tensor rand_like(Tensor t, torch::TensorOptions options, bool requires_grad) {
        return Tensor(torch::rand_like(t.data(), options), requires_grad);
    };

    Tensor randn(torch::IntArrayRef shape, torch::TensorOptions options, bool requires_grad) {
        return Tensor(torch::randn(shape, options), requires_grad);
    };

    Tensor randn_like(Tensor t, torch::TensorOptions options, bool requires_grad) {
        return Tensor(torch::randn_like(t.data(), options), requires_grad);
    };

    Tensor full(torch::IntArrayRef shape, double fill_value, torch::TensorOptions options, bool requires_grad) {
        return Tensor(torch::full(shape, fill_value, options), requires_grad);
    };

    Tensor full_like(Tensor t, double fill_value, torch::TensorOptions options, bool requires_grad) {
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
                
                any_grad = any_grad || t.requires_grad();
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
            return std::make_shared<autograd::AddBackward>(t1, t2);
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
            return std::make_shared<autograd::SubBackward>(t1, t2);
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
            return std::make_shared<autograd::MulBackward>(t1, t2);
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
            return std::make_shared<autograd::DivBackward>(t1, t2);
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
                return std::make_shared<autograd::MatmulBackward>(t1, t2);
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
            return std::make_shared<autograd::PowBackward>(base, exp, outputs[0]);
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

        NodeFactory node_factory = [&t, device](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::NotImplementedBackward>();
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor copy_to(const Tensor& t, torch::Dtype dtype) {
        TensorList inputs {t};

        Operation operation = [&t, dtype]() -> TorchList {
            return { t.data().to(dtype) };
        };

        NodeFactory node_factory = [&t, dtype](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::NotImplementedBackward>();
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }


    Tensor clone(const Tensor& t) {
        TensorList inputs {t};

        Operation operation = [&t]() -> TorchList {
            return { torch::clone(t.data()) };
        };

        NodeFactory node_factory = [&t](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::NotImplementedBackward>();
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
            return std::make_shared<autograd::NotImplementedBackward>();
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
            return std::make_shared<autograd::NotImplementedBackward>();
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
            return std::make_shared<autograd::NotImplementedBackward>();
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
            return std::make_shared<autograd::NotImplementedBackward>();
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
            return std::make_shared<autograd::NotImplementedBackward>();
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
            return std::make_shared<autograd::NotImplementedBackward>();
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
            return std::make_shared<autograd::NotImplementedBackward>();
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

        NodeFactory node_factory = [&t, dim, keepdim](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::NotImplementedBackward>();
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
            return std::make_shared<autograd::NotImplementedBackward>();
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

        NodeFactory node_factory = [&t, dim, keepdim](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::NotImplementedBackward>();
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];
        
        return {output, min_indices};
    }

    Tensor squeeze(const Tensor& t, torch::IntArrayRef dim) {
        TensorList inputs {t};

        Operation operation = [&t, dim]() -> TorchList {
            return { torch::squeeze(t.data(), dim) };
        };

        NodeFactory node_factory = [&t, dim](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::NotImplementedBackward>();
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor unsqueeze(const Tensor& t, int64_t dim) {
        TensorList inputs {t};

        Operation operation = [&t, dim]() -> TorchList {
            return { torch::unsqueeze(t.data(), dim) };
        };

        NodeFactory node_factory = [&t, dim](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::NotImplementedBackward>();
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    Tensor reshape(const Tensor& t, torch::IntArrayRef shape) {
        TensorList inputs {t};

        Operation operation = [&t, shape]() -> TorchList {
            return { torch::reshape(t.data(), shape) };
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

        NodeFactory node_factory = [&t, dim0, dim1](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::NotImplementedBackward>();
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

        NodeFactory node_factory = [&t, src, dest](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::NotImplementedBackward>();
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

        NodeFactory node_factory = [dim](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::NotImplementedBackward>();
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
            return std::make_shared<autograd::NotImplementedBackward>();
        };

        Tensor output = apply_operation(inputs, operation, node_factory)[0];

        return output;
    }

    TensorList unbind(const Tensor& t, int64_t dim) {
        const TensorList& inputs {t};

        Operation operation = [&t, dim]() -> TorchList {
            return torch::unbind(t.data(), dim);
        };

        NodeFactory node_factory = [dim](const TensorList& outputs) -> autograd::NodePtr {
            return std::make_shared<autograd::NotImplementedBackward>();
        };

        TensorList outputs = apply_operation(inputs, operation, node_factory);

        return outputs;
    }

} // namespace synapx
