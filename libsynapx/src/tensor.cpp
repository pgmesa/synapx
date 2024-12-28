
#include <torch/torch.h>
#include <iostream>

#include <synapx/tensor.hpp>


namespace synapx {

// Constructor implementation
Tensor::Tensor(const torch::Tensor& tensor) : data(tensor) {}

size_t Tensor::numel() const {
    return data.numel();
}

size_t Tensor::ndim() const {
    return data.dim();
}

} // namespace synapx

// template <typename T>
// std::vector<size_t> Tensor<T>::shape() const {
//     return std::vector<size_t>(array.shape().begin(), array.shape().end());
// }

// template <typename T>
// std::vector<size_t> Tensor<T>::strides() const {
//     return std::vector<size_t>(array.strides().begin(), array.strides().end());
// }

// template <typename T>
// Tensor<T> Tensor<T>::empty(const std::vector<size_t>& shape) {
//     return Tensor<T>(xt::empty<T>(shape));
// }

// template <typename T>
// Tensor<T> Tensor<T>::full(const std::vector<size_t>& shape, T value) {
//     auto array = xt::xarray<T>::from_shape(shape);
//     array.fill(value);
//     return Tensor<T>(array);
// }

// template <typename T>
// Tensor<T> Tensor<T>::zeros(const std::vector<size_t>& shape) {
//     return Tensor<T>(xt::zeros<T>(shape));
// }

// template <typename T>
// Tensor<T> Tensor<T>::ones(const std::vector<size_t>& shape) {
//     return Tensor<T>(xt::ones<T>(shape));
// }

// template <typename T>
// Tensor<T> Tensor<T>::view(const std::vector<size_t>& shape) const {
//     size_t total_elements = array.size();
//     size_t requested_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
//     if (total_elements != requested_elements) {
//         throw std::invalid_argument("Invalid shape for viewing: total elements do not match.");
//     }
//     auto viewed = xt::reshape_view(array, shape);
//     return Tensor<T>(viewed);
// }

// template <typename T>
// Tensor<T> Tensor<T>::broadcast_to(const std::vector<size_t>& shape) const {
//     auto broadcasted = xt::broadcast(array, shape);
//     return Tensor<T>(broadcasted);
// }

// template <typename T>
// Tensor<T> Tensor<T>::add(const Tensor<T>& other) const {
//     return F::add(*this, other);
// }

// template <typename T>
// Tensor<T> Tensor<T>::mul(const Tensor<T>& other) const {
//     return F::mul(*this, other);
// }

// // Matrix multiplication
// template <typename T>
// Tensor<T> Tensor<T>::matmul(const Tensor<T>& other) const {
//     return F::matmul(*this, other);
// }

// // template <typename T>
// // Tensor<T> Tensor<T>::matmul(const Tensor<T>& t1, const Tensor<T>& t2) {
// //     return F::matmul(t1, t2);
// // }

// template <typename T>
// template <typename U>
// Tensor<U> Tensor<T>::to() const {
//     if constexpr (std::is_same<T, U>::value) {
//         return *this;
//     }
//     auto casted = xt::cast<U>(array);
//     return Tensor<U>(casted);
// }

// // Print function
// template <typename T>
// void Tensor<T>::print() const {
//     std::cout << array << std::endl;
// }
