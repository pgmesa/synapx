
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

    bool t1_expanded = false;
    if (t1.numel < t2.numel) {
        t1 = t1.expand(t2.shape);
        t1_expanded = true;
    } else if (t2.numel < t1.numel) {
        t2 = t2.expand(t1.shape);
    }

    Tensor<T> out = inplace? t1: Tensor<T>::empty(t1.shape);
    if (inplace && t1_expanded) {
        throw std::runtime_error("Broadcasted Tensor doesn't support in-place operation");
    }
    cpu::add_forward(t1, t2, out);
    return out;
}

// template<typename T>
// Tensor<T> add(const Tensor<T>& t1, const double value) {
//     Tensor<T> t2 = Tensor<T>::full(t1.shape, value);
//     cpu::add_forward(t1.data.get(), t2.data.get(), t2.data.get(), t1.numel);
//     return out;
// }


} // namespace F


// Tensor Tensor::matmul(const Tensor& t2) const {
//     Tensor out = std::visit([this, &t2](auto& data_ptr) -> Tensor {
//         using T = std::decay_t<decltype(data_ptr.get()[0])>;
//         return Tensor::matmul_template<T>(*this, t2);
//     }, this->data);

//     return out;
// }

// Tensor Tensor::matmul(const Tensor& t1, const Tensor& t2) {
//     Tensor out = std::visit([&t1, &t2](auto& data_ptr) -> Tensor {
//         using T = std::decay_t<decltype(data_ptr.get()[0])>;
//         return Tensor::matmul_template<T>(t1, t2);
//     }, t1.data);

//     return out;
// }

// template<typename T>
// Tensor Tensor::matmul_template(const Tensor& t1, const Tensor& t2) {
//     if (t1.dtype != t2.dtype) {
//         throw std::invalid_argument("Datatypes must match.");
//     } 
//     if (t1.ndim < 2 || t2.ndim < 2) {
//         throw std::invalid_argument("Tensor dimensions must be greater than 1");
//     }
//     if (t1.shape[t1.ndim - 1] != t2.shape[t2.ndim - 2]) {
//         throw std::invalid_argument("Invalid Tensor dimensions");
//     }

//     int ndim = t1.ndim;
//     int out_t1_dim = t1.shape[ndim - 2];
//     int out_t2_dim = t2.shape[ndim - 1];
//     int last_t1_dim = t1.shape[ndim - 1];
//     int other_t2_dim = t2.shape[ndim - 2];
    
//     std::vector<int> out_shape(ndim);
//     out_shape[ndim - 2] = out_t1_dim;
//     out_shape[ndim - 1] = out_t2_dim;
//     for (int i=0; i < ndim - 2; i++) {
//         out_shape[i] = t1.shape[i];
//     }
//     Tensor out = Tensor::zeros(out_shape, t1.dtype);

//     T* t1_data = t1.get_data<T>();
//     T* t2_data = t2.get_data<T>();
//     T* out_data = out.get_data<T>();

//     size_t batches = t1.numel / out_t1_dim / last_t1_dim;
//     for (int k=0; k < batches; k++) {
//         for (int row=0; row < out_t1_dim; row++) {
//             for (int j=0; j < last_t1_dim; j++) {
//                 for (int i=0; i < out_t2_dim; i++) {
//                     int out_index = k*out_t1_dim*out_t2_dim + row*out_t2_dim + i;
//                     int t1_index = k*out_t1_dim*last_t1_dim + row*last_t1_dim + j;
//                     int t2_index = k*other_t2_dim*out_t2_dim + j*out_t2_dim + i;
                    
//                     out_data[out_index] += t1_data[t1_index] * t2_data[t2_index];
//                 }
//             }   
//         }
//     } 
//     return out;
// }

#endif