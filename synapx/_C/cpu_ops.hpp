#ifndef CPU_OPS_HPP
#define CPU_OPS_HPP

#include "tensor.hpp"

namespace cpu {

template<typename T>
void add_forward(T *p1, T *p2, T *pout, int length) {
    for (int i=0; i < length; i++) {
        pout[i] = p1[i] + p2[i];
    }
}

template<typename T>
void add_backward(Tensor<T> t1, T *t2, T *out, int length) {
    // TODO 
}

} // namespace cpu

#endif


