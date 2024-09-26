#ifndef CPU_OPS_HPP
#define CPU_OPS_HPP

#include "tensor.hpp"

namespace cpu {

template<typename T>
void add_forward(const Tensor<T>& t1, const Tensor<T>& t2, Tensor<T>& out) {
    for (int i=0; i < t1.numel; i++) {
        out.data[i] = t1.get(i) + t2.get(i);
    }
}

// template<typename T>
// void add_backward(Tensor<T> t1, T *t2, T *out, int length) {
//     // TODO 
// }

} // namespace cpu

#endif


