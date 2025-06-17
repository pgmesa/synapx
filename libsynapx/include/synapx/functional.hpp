#ifndef FUNCTIONAL_HPP
#define FUNCTIONAL_HPP

#include <tuple>
#include <memory>

#include <synapx/core.hpp>
#include <synapx/tensor.hpp>


namespace synapx {

    SYNAPX_API Tensor empty(torch::IntArrayRef shape, torch::TensorOptions options = {}, bool requires_grad = false);
    SYNAPX_API Tensor empty_like(Tensor t, torch::TensorOptions options = {}, bool requires_grad = false);
    SYNAPX_API Tensor ones(torch::IntArrayRef shape, torch::TensorOptions options = {}, bool requires_grad = false);
    SYNAPX_API Tensor ones_like(Tensor t, torch::TensorOptions options = {}, bool requires_grad = false);
    SYNAPX_API Tensor zeros(torch::IntArrayRef shape, torch::TensorOptions options = {}, bool requires_grad = false);
    SYNAPX_API Tensor zeros_like(Tensor t, torch::TensorOptions options = {}, bool requires_grad = false);
    SYNAPX_API Tensor rand(torch::IntArrayRef shape, torch::TensorOptions options = {}, bool requires_grad = false);
    SYNAPX_API Tensor rand_like(Tensor t, torch::TensorOptions options = {}, bool requires_grad = false);
    SYNAPX_API Tensor randn(torch::IntArrayRef shape, torch::TensorOptions options = {}, bool requires_grad = false);
    SYNAPX_API Tensor randn_like(Tensor t, torch::TensorOptions options = {}, bool requires_grad = false);
    SYNAPX_API Tensor full(torch::IntArrayRef shape, double fill_value, torch::TensorOptions options = {}, bool requires_grad = false);
    SYNAPX_API Tensor full_like(Tensor t, double fill_value, torch::TensorOptions options = {}, bool requires_grad = false);

    SYNAPX_API Tensor add(const Tensor& t1, const Tensor& t2);
    SYNAPX_API Tensor add(const Tensor& t1, double t2);
    SYNAPX_API Tensor sub(const Tensor& t1, const Tensor& t2);
    SYNAPX_API Tensor sub(const Tensor& t1, double t2);
    SYNAPX_API Tensor mul(const Tensor& t1, const Tensor& t2);
    SYNAPX_API Tensor mul(const Tensor& t1, double t2);
    SYNAPX_API Tensor div(const Tensor& t1, const Tensor& t2);
    SYNAPX_API Tensor div(const Tensor& t1, double t2);
    SYNAPX_API Tensor pow(const Tensor& base, const Tensor& exp);
    SYNAPX_API Tensor pow(const Tensor& base, double exp);
    SYNAPX_API Tensor matmul(const Tensor& t1, const Tensor& t2);
    SYNAPX_API Tensor neg(const Tensor& t);

    SYNAPX_API Tensor rsub(const Tensor& t1, const Tensor& t2);
    SYNAPX_API Tensor rsub(const Tensor& t1, double t2);
    SYNAPX_API Tensor rpow(const Tensor& t1, const Tensor& exp);
    SYNAPX_API Tensor rpow(const Tensor& t1, double exp);
    SYNAPX_API Tensor rdiv(const Tensor& t1, const Tensor& t2);
    SYNAPX_API Tensor rdiv(const Tensor& t1, double t2);
    SYNAPX_API Tensor rmatmul(const Tensor& t1, const Tensor& t2);

    SYNAPX_API Tensor copy_to(const Tensor& t, torch::Device device);
    SYNAPX_API Tensor copy_to(const Tensor& t, torch::Dtype dtype);
    SYNAPX_API Tensor clone(const Tensor& t);
    SYNAPX_API Tensor addmm(const Tensor& inp, const Tensor& mat1, const Tensor& mat2);
    SYNAPX_API Tensor exp(const Tensor& t);
    SYNAPX_API Tensor log(const Tensor& t);
    SYNAPX_API Tensor sqrt(const Tensor& t);
    SYNAPX_API Tensor sum(const Tensor& t, torch::IntArrayRef dim = {}, bool keepdim = false);
    SYNAPX_API Tensor mean(const Tensor& t, torch::IntArrayRef dim = {}, bool keepdim = false);
    SYNAPX_API Tensor max(const Tensor& t);
    SYNAPX_API std::tuple<Tensor, Tensor> max(const Tensor& t, int64_t dim, bool keepdim = false);
    SYNAPX_API Tensor min(const Tensor& t);
    SYNAPX_API std::tuple<Tensor, Tensor> min(const Tensor& t, int64_t dim, bool keepdim = false);
    SYNAPX_API Tensor squeeze(const Tensor& t, torch::IntArrayRef dim = {});
    SYNAPX_API Tensor unsqueeze(const Tensor& t, int64_t dim);
    SYNAPX_API Tensor reshape(const Tensor& t, torch::IntArrayRef shape);
    SYNAPX_API Tensor broadcast_to(const Tensor& t, torch::IntArrayRef shape);
    SYNAPX_API Tensor transpose(const Tensor& t, int64_t dim0, int64_t dim1);
    SYNAPX_API Tensor swapdims(const Tensor& t, int64_t dim0, int64_t dim1);
    SYNAPX_API Tensor movedim(const Tensor& t, int64_t src, int64_t dest);
    SYNAPX_API Tensor slice(const Tensor& t, const TensorIndices& indices);
    SYNAPX_API Tensor concat(const TensorList& tensors, int64_t dim = 0);
    SYNAPX_API Tensor stack(const TensorList& tensors, int64_t dim = 0);
    SYNAPX_API TensorList unbind(const Tensor& t, int64_t dim = 0);

}

#endif // FUNCTIONAL_HPP