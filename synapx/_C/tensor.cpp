
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"    // for printing xtensor arrays
#include "xtensor/xadapt.hpp"
#include "cblas.h
#include <iostream>
#include <vector>


template <typename T>
class Tensor {
private:
    xt::xarray<T> tensor;  // xtensor's multidimensional array

public:
    // Constructor: Initialize with dimensions as std::vector
    Tensor(const std::vector<size_t>& dims) : tensor(dims) {}

    // Constructor: Initialize with a predefined tensor
    Tensor(const xt::xarray<T>& arr) : tensor(arr) {}

    // Static method to create a tensor full of zeros
    static Tensor<T> zeros(const std::vector<size_t>& dims) {
        return Tensor<T>(xt::zeros<T>(dims));
    }

    // Static method to create a tensor full of ones
    static Tensor<T> ones(const std::vector<size_t>& dims) {
        return Tensor<T>(xt::ones<T>(dims));
    }

    // Static method to create a tensor full of a specific value
    // static Tensor<T> full(const std::vector<size_t>& dims, T value) {
    //     return Tensor<T>(xt::xfull<T>(dims, value));
    // }

    // Set value at a specific index
    // void setValue(const std::vector<size_t>& indices, T value) {
    //     xt::xindex index(indices.begin(), indices.end());  // Convert vector to xt::xindex
    //     tensor(index) = value;
    // }

    // // Get value at a specific index
    // T getValue(const std::vector<size_t>& indices) const {
    //     xt::xindex index(indices.begin(), indices.end());  // Convert vector to xt::xindex
    //     return tensor(index);
    // }

    // Add operation: element-wise addition of two tensors
    Tensor<T> add(const Tensor<T>& other) const {
        std::vector<size_t> result_shape(tensor.shape().begin(), tensor.shape().end());
        Tensor<T> result(result_shape);
        result.tensor = tensor + other.tensor;
        return result;
    }

    // Matrix multiplication using OpenBLAS
    Tensor<T> matmul(const Tensor<T>& other) const {
        auto shape_a = tensor.shape();
        auto shape_b = other.tensor.shape();

        // Case 1: Both tensors are 2D
        if (shape_a.size() == 2 && shape_b.size() == 2) {
            size_t M = shape_a[0];  // rows in A
            size_t K = shape_a[1];  // columns in A and rows in B
            size_t N = shape_b[1];  // columns in B

            if (K != shape_b[0]) {
                throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");
            }

            // Resulting matrix will have shape (M, N)
            Tensor<T> result(std::vector<size_t>{M, N});
            const xt::xarray<T>& A = this->tensor;   // Note the const-correctness
            const xt::xarray<T>& B = other.tensor;   // Use const reference
            xt::xarray<T>& C = result.tensor;

            // Perform matrix multiplication: C = A * B
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C.data(), N);

            return result;
        }
        // Case 2: Both tensors have more than 2 dimensions (batch matmul)
        else if (shape_a.size() > 2 && shape_b.size() > 2) {
            // Ensure last two dimensions are valid for matmul
            size_t K = shape_a[shape_a.size() - 1];
            size_t N = shape_b[shape_b.size() - 1];
            size_t M = shape_a[shape_a.size() - 2];

            if (K != shape_b[shape_b.size() - 2]) {
                throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");
            }

            // Flatten the batch dimensions
            size_t batch_size = 1;
            for (size_t i = 0; i < shape_a.size() - 2; ++i) {
                batch_size *= shape_a[i];
            }

            // Resulting tensor will have shape (batch_size, M, N)
            std::vector<size_t> result_shape(shape_a.begin(), shape_a.end());
            result_shape[result_shape.size() - 2] = M;
            result_shape[result_shape.size() - 1] = N;
            Tensor<T> result(result_shape);

            const xt::xarray<T>& A = this->tensor;   // Use const reference
            const xt::xarray<T>& B = other.tensor;   // Use const reference
            xt::xarray<T>& C = result.tensor;

            // Perform matrix multiplication for each batch
            for (size_t batch = 0; batch < batch_size; ++batch) {
                // Get pointers to the submatrices for the current batch
                const T* A_ptr = A.data() + batch * M * K;
                const T* B_ptr = B.data() + batch * K * N;
                T* C_ptr = C.data() + batch * M * N;

                // Perform matrix multiplication: C = A * B
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            M, N, K, 1.0f, A_ptr, K, B_ptr, N, 0.0f, C_ptr, N);
            }

            return result;
        } else {
            throw std::invalid_argument("Unsupported tensor dimensions for matmul.");
        }
    }

    // Print tensor (for demonstration purposes)
    void print() const {
        std::cout << tensor << std::endl;
    }
};