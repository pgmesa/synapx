#ifndef AUTOGRAD_FUNCTIONS_HPP
#define AUTOGRAD_FUNCTIONS_HPP

#include <synapx/autograd/graph.hpp>

#include <stdexcept>

#include <fmt/core.h>
#include <synapx/functional.hpp>


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

    class ToCopyBackward0: public Node {
    public:
        ToCopyBackward0(torch::Device);
        ToCopyBackward0(torch::Dtype);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        std::optional<torch::Device> device;
        std::optional<torch::Dtype> dtype;
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
        ExpBackward0(const Tensor& fw_result);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        Tensor fw_result;
    };


    class LogBackward0: public Node {
    public:
        LogBackward0(const Tensor& t);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        Tensor t;
    };


    class SqrtBackward0: public Node {
    public:
        SqrtBackward0(const Tensor& fw_result);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        Tensor fw_result;
    };


    class SumBackward0: public Node {
    public:
        SumBackward0(const Tensor& t, torch::IntArrayRef dim, bool keepdim);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        IntArray t_shape;
        IntArray dim;
        bool keepdim;
    };


    class MeanBackward0: public Node {
    public:
        MeanBackward0(const Tensor& t, torch::IntArrayRef dim, bool keepdim);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        IntArray t_shape;
        IntArray dim;
        bool keepdim;
        IntArray normalized_dims;
    };


    class MaxBackward0: public Node {
    public:
        MaxBackward0(const Tensor& t, int64_t dim, bool keepdim, const Tensor& max_indices);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        IntArray t_shape;
        int64_t dim;
        bool keepdim;
        const Tensor max_indices;
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
        IntArray t_shape;
        int64_t dim;
        bool keepdim;
        const Tensor min_indices;
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


    class SqueezeBackward0: public Node {
    public:
        SqueezeBackward0(const Tensor& t, torch::IntArrayRef dim);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        IntArray t_shape;
        IntArray dim;
    };


    class UnsqueezeBackward0: public Node {
    public:
        UnsqueezeBackward0(int64_t dim);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        int64_t dim;
    };


    class ReshapeBackward0: public Node {
    public:
        ReshapeBackward0(const Tensor& t);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        IntArray t_shape;
    };


    class TransposeBackward0: public Node {
    public:
        TransposeBackward0(int64_t dim0, int64_t dim1);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        int64_t dim0;
        int64_t dim1;
    };


    class MovedimBackward0: public Node {
    public:
        MovedimBackward0(int64_t src, int64_t dest);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        int64_t src;
        int64_t dest;
    };


    class SliceBackward0: public Node {
    public:
        SliceBackward0(const Tensor& t, const TensorIndices& indices);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        IntArray t_shape;
        TensorIndices indices;
    };


    class ConcatBackward0: public Node {
    public:
        ConcatBackward0(const TensorList& inputs, int64_t dim);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        int64_t dim;
        IntArray sizes;
    };


    class StackBackward0: public Node {
    public:
        StackBackward0(int64_t dim);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        int64_t dim;
    };


    class UnbindBackward0: public Node {
    public:
        UnbindBackward0(const Tensor& t, int64_t dim);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        IntArray t_shape;
        int64_t dim;
    };


    // Activations
    class ReLUBackward0: public Node {
    public:
        ReLUBackward0(const Tensor& t);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        Tensor t;
    };

    class SigmoidBackward0: public Node {
    public:
        SigmoidBackward0(const Tensor& fw_result);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        Tensor fw_result;
    };

    class SoftmaxBackward0: public Node {
    public:
        SoftmaxBackward0(const Tensor& fw_result, int64_t dim);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        Tensor fw_result;
        int64_t dim;
    };

    class LogSoftmaxBackward0: public Node {
    public:
        LogSoftmaxBackward0(const Tensor& fw_result, int64_t dim);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        Tensor fw_result;
        int64_t dim;
    };


    // Losses
    class MSELossBackward0: public Node {
    public:
        MSELossBackward0(const Tensor& input, const Tensor& target, const Tensor& diff, Reduction reduction);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        bool input_req_grad;
        bool target_req_grad;
        Tensor diff;
        Reduction reduction;
    };

    class NLLLossBackward0: public Node {
    public:
        NLLLossBackward0(const Tensor& input, const Tensor& target, Reduction reduction);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        Tensor input;
        Tensor target;
        Reduction reduction;
    };


} // namespace synapx::autograd

#endif
