#ifndef AUTOGRAD_FUNCTIONS_HPP
#define AUTOGRAD_FUNCTIONS_HPP

#include <synapx/autograd/graph.hpp>


namespace synapx::autograd {

    class NotImplementedBackward: public Node {
    public:
        std::string name() const override { return "NotImplementedBackward"; };
        TensorList apply(const TensorList& inputs) override { return {}; };
    };

    
    class AccumulateGrad: public Node {
    public:
        AccumulateGrad(const Tensor& variable);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;
    
    private:
        Tensor variable;
    };


    class AddBackward: public Node {
    public:
        AddBackward(const Tensor& t1, const Tensor& t2);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        bool t1_req_grad;
        bool t2_req_grad;
        IntArray t1_shape;
        IntArray t2_shape;
    };


    class SubBackward: public Node {
    public:
        SubBackward(const Tensor& t1, const Tensor& t2);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        bool t1_req_grad;
        bool t2_req_grad;
        IntArray t1_shape;
        IntArray t2_shape;
    };


    class MulBackward: public Node {
    public:
        MulBackward(const Tensor& t1, const Tensor& t2);
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


    class DivBackward: public Node {
    public:
        DivBackward(const Tensor& t1, const Tensor& t2);
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


    class MatmulBackward: public Node {
    public:
        MatmulBackward(const Tensor& t1, const Tensor& t2);
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


    class PowBackward: public Node {
    public:
        PowBackward(const Tensor& base, const Tensor& exp, const Tensor& fw_result);
        std::string name() const override;
        TensorList apply(const TensorList& inputs) override;

    private:
        bool base_req_grad;
        bool exp_req_grad;
        Tensor base;
        Tensor exp;
        Tensor fw_result;
    };


} // namespace synapx::autograd

#endif
