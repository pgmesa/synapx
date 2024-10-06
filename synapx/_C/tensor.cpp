
#include "tensor.hpp"
#include "functional.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xstrided_view.hpp"
#include "cblas.h"
#include <iostream>


// Explicit template instantiations
template class Tensor<float>;
template class Tensor<int32_t>;
template class Tensor<uint8_t>;

template Tensor<int32_t> Tensor<float>::to<int32_t>() const;
template Tensor<uint8_t> Tensor<float>::to<uint8_t>() const;
template Tensor<float> Tensor<int32_t>::to<float>() const;
template Tensor<uint8_t> Tensor<int32_t>::to<uint8_t>() const;
template Tensor<float> Tensor<uint8_t>::to<float>() const;
template Tensor<int32_t> Tensor<uint8_t>::to<int32_t>() const;

// Constructor implementations
template <typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shape) : array(shape) {}

template <typename T>
Tensor<T>::Tensor(const xt::xarray<T>& arr) : array(arr) {}

template <typename T>
size_t Tensor<T>::numel() const {
    return array.size();
}

template <typename T>
size_t Tensor<T>::ndim() const {
    return array.shape().size();
}

template <typename T>
std::vector<size_t> Tensor<T>::shape() const {
    return std::vector<size_t>(array.shape().begin(), array.shape().end());
}

template <typename T>
std::vector<size_t> Tensor<T>::strides() const {
    return std::vector<size_t>(array.strides().begin(), array.strides().end());
}

template <typename T>
Tensor<T> Tensor<T>::empty(const std::vector<size_t>& shape) {
    return Tensor<T>(xt::empty<T>(shape));
}

template <typename T>
Tensor<T> Tensor<T>::full(const std::vector<size_t>& shape, T value) {
    auto array = xt::xarray<T>::from_shape(shape);
    array.fill(value);
    return Tensor<T>(array);
}

template <typename T>
Tensor<T> Tensor<T>::zeros(const std::vector<size_t>& shape) {
    return Tensor<T>(xt::zeros<T>(shape));
}

template <typename T>
Tensor<T> Tensor<T>::ones(const std::vector<size_t>& shape) {
    return Tensor<T>(xt::ones<T>(shape));
}

template <typename T>
Tensor<T> Tensor<T>::view(const std::vector<size_t>& shape) const {
    size_t total_elements = array.size();
    size_t requested_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    if (total_elements != requested_elements) {
        throw std::invalid_argument("Invalid shape for viewing: total elements do not match.");
    }
    auto viewed = xt::reshape_view(array, shape);
    return Tensor<T>(viewed);
}

template <typename T>
Tensor<T> Tensor<T>::broadcast_to(const std::vector<size_t>& shape) const {
    auto broadcasted = xt::broadcast(array, shape);
    return Tensor<T>(broadcasted);
}

template <typename T>
Tensor<T> Tensor<T>::add(const Tensor<T>& other) const {
    return F::add(*this, other);
}

template <typename T>
Tensor<T> Tensor<T>::mul(const Tensor<T>& other) const {
    return F::mul(*this, other);
}

// Matrix multiplication
template <typename T>
Tensor<T> Tensor<T>::matmul(const Tensor<T>& other) const {
    return F::matmul(*this, other);
}

template <typename T>
Tensor<T> Tensor<T>::matmul(const Tensor<T>& t1, const Tensor<T>& t2) {
    return F::matmul(t1, t2);
}

template <typename T>
template <typename U>
Tensor<U> Tensor<T>::to() const {
    if constexpr (std::is_same<T, U>::value) {
        return *this;
    }
    auto casted = xt::cast<U>(array);
    return Tensor<U>(casted);
}

// Print function
template <typename T>
void Tensor<T>::print() const {
    std::cout << array << std::endl;
}
