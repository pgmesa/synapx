#ifndef FUNCTIONAL_HPP
#define FUNCTIONAL_HPP

#include "tensor.hpp"
#include "cpu_ops.hpp"
#include "utils.hpp"


namespace F {

template<typename T>
Tensor<T> add(const Tensor<T>& t1, const Tensor<T>& t2) {
    auto result = cpu::add_forward(t1.array, t2.array);
    Tensor<T> out(result);
    return out;
}


template<typename T>
Tensor<T> mul(const Tensor<T>& t1, const Tensor<T>& t2) {
    auto result = cpu::mul_forward(t1.array, t2.array);
    Tensor<T> out(result);
    return out;
}


template<typename T>
Tensor<T> matmul(Tensor<T> t1, Tensor<T> t2) {
    auto [t1_shape, t2_shape] = utils::broadcast_shapes_for_matmul(t1.shape(), t2.shape());

    size_t t1_numel = std::accumulate(t1_shape.begin(), t1_shape.end(), 1, std::multiplies<size_t>());
    if (t1.numel() == t1_numel) {
        t1 = t1.view(t1_shape);
    } else {
        t1 = t1.broadcast_to(t1_shape);
    }

    size_t t2_numel = std::accumulate(t2_shape.begin(), t2_shape.end(), 1, std::multiplies<size_t>());
    if (t2.numel() == t2_numel) {
        t2 = t2.view(t2_shape);
    } else {
        t2 = t2.broadcast_to(t2_shape);
    }
    
    size_t nbatches = std::accumulate(t1_shape.begin(), t1_shape.end() - 2, 1, std::multiplies<size_t>());
    t1 = t1.view({nbatches, t1_shape[t1_shape.size() - 2], t1_shape.back()});
    t2 = t2.view({nbatches, t2_shape[t2_shape.size() - 2], t2_shape.back()});

    Tensor<float> t1f = t1.template to<float>();
    Tensor<float> t2f = t2.template to<float>();
    xt::xarray<float> resultf = cpu::bmatmul_forward(t1f.array, t2f.array);
    xt::xarray<T> result = xt::cast<T>(resultf);

    std::vector<int> out_shape(t1_shape.begin(), t1_shape.end() - 1);
    out_shape.push_back(t2_shape.back());

    result = xt::reshape_view(result, out_shape);
    Tensor<T> out(result);

    return out;
}

} // namespace

#endif