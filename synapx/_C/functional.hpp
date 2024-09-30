
#ifndef FUNCTIONAL_HPP
#define FUNCTIONAL_HPP

#include "tensor.hpp"
#include "cpu_ops.hpp"

namespace F {

template<typename T>
Tensor<T> add(const Tensor<T>& t1_, const Tensor<T>& t2_, bool inplace=false) {
    Tensor<T> t1 = t1_;  // Create a copy of t1
    Tensor<T> t2 = t2_;  // Create a copy of t2

    if (t1.dtype != t2.dtype) {
        throw std::invalid_argument("Datatypes must match.");
    }

    std::vector<int> out_shape = utils::broadcast_shapes(t1.shape, t2.shape);

    t2 = t2.broadcast_to(out_shape);
    Tensor<T> out;
    if (inplace) {
        out = t1;
    } else {
        t1 = t1.broadcast_to(out_shape);
        out = Tensor<T>::empty(out_shape);
    }

    if (!utils::shapes_equal(out.shape, t2.shape)) {
        std::string out_shape_str = utils::vector_to_string(out.shape);
        std::string t2_shape_str = utils::vector_to_string(t2.shape);
        std::string err_msg = "Output with shape " + out_shape_str + 
                                " doesn't match the broadcast shape " + t2_shape_str;
        throw std::runtime_error(err_msg);
    }

    cpu::add_forward(t1, t2, out.data.get());
    return out;
}

template<typename T>
Tensor<T> mul(const Tensor<T>& t1_, const Tensor<T>& t2_, bool inplace=false) {
    Tensor<T> t1 = t1_;  // Create a copy of t1
    Tensor<T> t2 = t2_;  // Create a copy of t2

    if (t1.dtype != t2.dtype) {
        throw std::invalid_argument("Datatypes must match.");
    }

    std::vector<int> out_shape = utils::broadcast_shapes(t1.shape, t2.shape);

    t2 = t2.broadcast_to(out_shape);
    Tensor<T> out;
    if (inplace) {
        out = t1;
    } else {
        t1 = t1.broadcast_to(out_shape);
        out = Tensor<T>::empty(out_shape);
    }

    if (!utils::shapes_equal(out.shape, t2.shape)) {
        std::string out_shape_str = utils::vector_to_string(out.shape);
        std::string t2_shape_str = utils::vector_to_string(t2.shape);
        std::string err_msg = "Output with shape " + out_shape_str + 
                                " doesn't match the broadcast shape " + t2_shape_str;
        throw std::runtime_error(err_msg);
    }

    cpu::mul_forward(t1, t2, out.data.get());
    return out;
}

template<typename T>
Tensor<T> matmul(const Tensor<T>& t1_, const Tensor<T>& t2_) {
    Tensor<T> t1 = t1_;  // Create a copy of t1
    Tensor<T> t2 = t2_;  // Create a copy of t2

    if (t1.dtype != t2.dtype) {
        throw std::invalid_argument("Datatypes must match.");
    }

    auto [t1_shape, t2_shape] = utils::broadcast_shapes_for_matmul(t1.shape, t2.shape);

    int t1_numel = std::accumulate(t1_shape.begin(), t1_shape.end(), 1, std::multiplies<int>());
    if (t1.numel == t1_numel) {
        t1 = t1.view(t1_shape);
    } else {
        t1 = t1.broadcast_to(t1_shape);
    }

    int t2_numel = std::accumulate(t2_shape.begin(), t2_shape.end(), 1, std::multiplies<int>());
    if (t2.numel == t2_numel) {
        t2 = t2.view(t2_shape);
    } else {
        t2 = t2.broadcast_to(t2_shape);
    }

    std::vector<int> out_shape(t1_shape.begin(), t1_shape.end() - 1);
    out_shape.push_back(t2_shape.back());

    Tensor<T> out = Tensor<T>::zeros(out_shape);
    cpu::matmul_forward(t1, t2, out.data.get());
    return out;
}

} // namespace F

#endif