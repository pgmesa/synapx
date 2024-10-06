#ifndef CPU_OPS_HPP
#define CPU_OPS_HPP

#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"
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

xt::xarray<float> bmatmul_forward(const xt::xarray<float>& a, const xt::xarray<float>& b) {
    // Get shapes of the input tensors
    const auto& a_shape = a.shape();
    const auto& b_shape = b.shape();
    
    // a and b have the same number of batches, and both are of size 3
    size_t batch_size = a_shape[0];
    size_t m = a_shape[1];   // Rows of matrix A in each batch
    size_t k = a_shape[2];   // Columns of matrix A in each batch, should be equal to rows of matrix B
    size_t n = b_shape[2];   // Columns of matrix B in each batch

    // Ensure that the inner dimensions match
    if (k != b_shape[1]) {
        throw std::invalid_argument("Inner dimensions do not match for matrix multiplication.");
    }

    // Create output tensor with the correct shape [batch_size, m, n]
    std::vector<size_t> out_shape = {batch_size, m, n};
    auto result = xt::xarray<float>::from_shape(out_shape);

    // Iterate over each batch and perform matrix multiplication
    for (size_t i = 0; i < batch_size; ++i) {
        const float* a_ptr = a.data() + i * m * k;
        const float* b_ptr = b.data() + i * k * n;
        float* c_ptr = result.data() + i * m * n;

        // Perform matrix multiplication
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    m, n, k, 
                    1.0f, a_ptr, k, 
                    b_ptr, n, 
                    0.0f, c_ptr, n);
    }

    return result;
}

// auto shape_a = array.shape();
//     auto shape_b = other.array.shape();

//     // Case 1: Both tensors are 2D
//     if (shape_a.size() == 2 && shape_b.size() == 2) {
//         size_t M = shape_a[0];  // rows in A
//         size_t K = shape_a[1];  // columns in A and rows in B
//         size_t N = shape_b[1];  // columns in B

//         if (K != shape_b[0]) {
//             throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");
//         }

//         // Resulting matrix will have shape (M, N)
//         Tensor<T> result(std::vector<size_t>{M, N});
//         const xt::xarray<T>& A = this->array;   // Note the const-correctness
//         const xt::xarray<T>& B = other.array;   // Use const reference
//         xt::xarray<T>& C = result.array;

//         // Perform matrix multiplication: C = A * B
//         cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
//                     M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C.data(), N);

//         return result;
//     }
//     // Case 2: Both tensors have more than 2 dimensions (batch matmul)
//     else if (shape_a.size() > 2 && shape_b.size() > 2) {
//         // Ensure last two dimensions are valid for matmul
//         size_t K = shape_a[shape_a.size() - 1];
//         size_t N = shape_b[shape_b.size() - 1];
//         size_t M = shape_a[shape_a.size() - 2];

//         if (K != shape_b[shape_b.size() - 2]) {
//             throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");
//         }

//         // Flatten the batch dimensions
//         size_t batch_size = 1;
//         for (size_t i = 0; i < shape_a.size() - 2; ++i) {
//             batch_size *= shape_a[i];
//         }

//         // Resulting array will have shape (batch_size, M, N)
//         std::vector<size_t> result_shape(shape_a.begin(), shape_a.end());
//         result_shape[result_shape.size() - 2] = M;
//         result_shape[result_shape.size() - 1] = N;
//         Tensor<T> result(result_shape);

//         const xt::xarray<T>& A = this->array;   // Use const reference
//         const xt::xarray<T>& B = other.array;   // Use const reference
//         xt::xarray<T>& C = result.array;

//         // Perform matrix multiplication for each batch
//         for (size_t batch = 0; batch < batch_size; ++batch) {
//             // Get pointers to the submatrices for the current batch
//             const T* A_ptr = A.data() + batch * M * K;
//             const T* B_ptr = B.data() + batch * K * N;
//             T* C_ptr = C.data() + batch * M * N;

//             // Perform matrix multiplication: C = A * B
//             cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
//                         M, N, K, 1.0f, A_ptr, K, B_ptr, N, 0.0f, C_ptr, N);
//         }

//         return result;
//     } else {
//         throw std::invalid_argument("Unsupported array dimensions for matmul.");
//     }

} // namespace

#endif