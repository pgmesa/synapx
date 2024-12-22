#ifndef CPU_OPS_HPP
#define CPU_OPS_HPP

#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xview.hpp"
#include "cblas.h"

namespace cpu {

template<typename T>
xt::xarray<T> add_forward(const xt::xarray<T>& a, const xt::xarray<T>& b) {
    return a + b;
}

template<typename T>
xt::xarray<T> mul_forward(const xt::xarray<T>& a, const xt::xarray<T>& b) {
    return a * b;
}

xt::xarray<float> bmatmul_forward(const xt::xarray<float>& a_3d, const xt::xarray<float>& b_3d) {
    size_t batch_size = a_3d.shape()[0];
    size_t m = a_3d.shape()[1];
    size_t k = a_3d.shape()[2];
    size_t n = b_3d.shape()[2];
    
    xt::xarray<float> result = xt::zeros<float>({batch_size, m, n});
    
    // Ensure contiguous memory
    xt::xarray<float> a_slice_storage(std::vector<size_t>{m, k});
    xt::xarray<float> b_slice_storage(std::vector<size_t>{k, n});
    
    for(size_t i = 0; i < batch_size; ++i) {
        // Get current slices
        auto a_slice = xt::view(a_3d, i, xt::all(), xt::all());
        auto b_slice = xt::view(b_3d, i, xt::all(), xt::all());
        
        // Copy to contiguous storage
        a_slice_storage = a_slice;
        b_slice_storage = b_slice;
        
        float* c_ptr = result.data() + i * m * n;
        
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k,
                    1.0f, a_slice_storage.data(), k,
                    b_slice_storage.data(), n,
                    0.0f, c_ptr, n);
    }
    
    return result;
}

} // namespace

#endif