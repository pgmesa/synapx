#ifndef CPU_OPS_HPP
#define CPU_OPS_HPP

#include "tensor.hpp"

namespace cpu {

template<typename T>
void add_forward(const Tensor<T>& t1, const Tensor<T>& t2, T* out) {
    for (int i=0; i < t1.numel; i++) {
        out[i] = t1.get(i) + t2.get(i);
    }
}

template<typename T>
void mul_forward(const Tensor<T>& t1, const Tensor<T>& t2, T* out) {
    for (int i=0; i < t1.numel; i++) {
        out[i] = t1.get(i) * t2.get(i);
    }
}

template<typename T>
void matmul_forward(const Tensor<T>& t1, const Tensor<T>& t2, T* out) {
    int dim_count = t1.ndim;
    int rows = t1.shape[dim_count - 2];
    int cols = t2.shape[dim_count - 1];
    int common_dim = t1.shape[dim_count - 1];

    size_t batch_size = t1.numel / (rows * common_dim);
    for (int batch = 0; batch < batch_size; batch++) {
        for (int i = 0; i < rows; i++) {
            for (int k = 0; k < common_dim; k++) {
                for (int j = 0; j < cols; j++) {
                    int out_idx = batch * rows * cols + i * cols + j;
                    int t1_idx = batch * rows * common_dim + i * common_dim + k;
                    int t2_idx = batch * common_dim * cols + k * cols + j;
                    
                    out[out_idx] += t1.get(t1_idx) * t2.get(t2_idx);
                }
            }   
        }
    }
}

} // namespace cpu

#endif


