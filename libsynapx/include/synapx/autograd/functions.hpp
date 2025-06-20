#ifndef AUTOGRAD_FUNCTIONS_HPP
#define AUTOGRAD_FUNCTIONS_HPP

#include <synapx/autograd/graph.hpp>

#include <stdexcept>

#include <fmt/core.h>


namespace synapx::autograd {

    class NotImplementedBackward: public Node {
    public:
        std::string name() const override { return "NotImplementedBackward"; };
        TensorList apply(const TensorList& inputs) override {
            throw std::runtime_error(fmt::format(
                "{}: Attempted to perform backward on an operation that does not implement a backward pass",
                name()
            ));
        }
    };

    
    class AccumulateGrad: public Node {
    public:
        AccumulateGrad(const Tensor& variable);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;
    
    private:
        Tensor variable;
    };


    class AddBackward0: public Node {
    public:
        AddBackward0(const Tensor& t1, const Tensor& t2);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        bool t1_req_grad;
        bool t2_req_grad;
        IntArray t1_shape;
        IntArray t2_shape;
    };


    class SubBackward0: public Node {
    public:
        SubBackward0(const Tensor& t1, const Tensor& t2);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        bool t1_req_grad;
        bool t2_req_grad;
        IntArray t1_shape;
        IntArray t2_shape;
    };


    class MulBackward0: public Node {
    public:
        MulBackward0(const Tensor& t1, const Tensor& t2);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        bool t1_req_grad;
        bool t2_req_grad;
        IntArray t1_shape;
        IntArray t2_shape;
        Tensor t1;
        Tensor t2;
    };


    class DivBackward0: public Node {
    public:
        DivBackward0(const Tensor& t1, const Tensor& t2);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        bool t1_req_grad;
        bool t2_req_grad;
        IntArray t1_shape;
        IntArray t2_shape;
        Tensor t1;
        Tensor t2;
    };


    class MatmulBackward0: public Node {
    public:
        MatmulBackward0(const Tensor& t1, const Tensor& t2);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        bool t1_req_grad;
        bool t2_req_grad;
        IntArray t1_shape;
        IntArray t2_shape;
        Tensor t1;
        Tensor t2;
    };


    class PowBackward0: public Node {
    public:
        PowBackward0(const Tensor& base, const Tensor& exp, const Tensor& fw_result);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        bool base_req_grad;
        bool exp_req_grad;
        Tensor base;
        Tensor fw_result;
        Tensor exp;
    };


    class CloneBackward0: public Node {
    public:
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        bool t_req_grad;
    };


    class AddmmBackward0: public Node {
    public:
        AddmmBackward0(const Tensor& inp, const Tensor& mat1, const Tensor& mat2);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        bool inp_req_grad;
        bool mat1_req_grad;
        bool mat2_req_grad;
        IntArray inp_shape;
        Tensor mat1;
        Tensor mat2;
    };


    class ExpBackward0: public Node {
    public:
        ExpBackward0(const Tensor& t, const Tensor& fw_result);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        bool t_req_grad;
        Tensor fw_result;
    };


    class LogBackward0: public Node {
    public:
        LogBackward0(const Tensor& t);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        bool t_req_grad;
        Tensor t;
    };


    class SqrtBackward0: public Node {
    public:
        SqrtBackward0(const Tensor& t, const Tensor& fw_result);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        bool t_req_grad;
        Tensor fw_result;
    };


    class SumBackward0: public Node {
    public:
        SumBackward0(const Tensor& t, const torch::IntArrayRef& dim, bool keepdim);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        bool t_req_grad;
        IntArray dim;
        bool keepdim;
        IntArray t_shape;
    };


    class MeanBackward0: public Node {
    public:
        MeanBackward0(const Tensor& t, const torch::IntArrayRef& dim, bool keepdim);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        bool t_req_grad;
        IntArray dim;
        bool keepdim;
        IntArray t_shape;
        IntArray normalized_dims;
    };


    class MaxBackward0: public Node {
    public:
        MaxBackward0(const Tensor& t, int64_t dim, bool keepdim, const Tensor& max_indices);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        bool t_req_grad;
        int64_t dim;
        bool keepdim;
        const Tensor max_indices;
        IntArray t_shape;
    };

    class MaxBackward1: public Node {
    public:
        MaxBackward1(const Tensor& t, const Tensor& max_value);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        Tensor t;
        Tensor max_value;
    };


    class MinBackward0: public Node {
    public:
        MinBackward0(const Tensor& t, int64_t dim, bool keepdim, const Tensor& min_indices);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        bool t_req_grad;
        int64_t dim;
        bool keepdim;
        const Tensor min_indices;
        IntArray t_shape;
    };

    class MinBackward1: public Node {
    public:
        MinBackward1(const Tensor& t, const Tensor& min_value);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        Tensor t;
        Tensor min_value;
    };


} // namespace synapx::autograd

#endif
