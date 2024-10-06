#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "xtensor/xarray.hpp"
#include <vector>


template<typename T>
class Tensor;
// Type trait to check for allowed types
template<typename T>
struct is_allowed_tensor_type : std::false_type {};
template<> struct is_allowed_tensor_type<uint8_t> : std::true_type {};
template<> struct is_allowed_tensor_type<int> : std::true_type {};
template<> struct is_allowed_tensor_type<float> : std::true_type {};


template <typename T>
class Tensor {
    static_assert(is_allowed_tensor_type<T>::value, "Tensor only supports uint8, int32, and float32");
public:
    xt::xarray<T> array;  // xtensor's multidimensional array

    // Constructors
    Tensor(const std::vector<size_t>& shape);
    Tensor(const xt::xarray<T>& arr);

    // Member functions
    size_t numel() const;
    size_t ndim() const;
    std::vector<size_t> shape() const;
    std::vector<size_t> strides() const;
    

    static Tensor<T> empty(const std::vector<size_t>& shape);
    static Tensor<T> full(const std::vector<size_t>& shape, T value);
    static Tensor<T> zeros(const std::vector<size_t>& shape);
    static Tensor<T> ones(const std::vector<size_t>& shape);

    Tensor<T> view(const std::vector<size_t>& shape) const;
    Tensor<T> broadcast_to(const std::vector<size_t>& shape) const;

    Tensor<T> add(const Tensor<T>& other) const;
    Tensor<T> mul(const Tensor<T>& other) const;
    Tensor<T> matmul(const Tensor<T>& other) const;
    static Tensor<T> matmul(const Tensor<T>& t1, const Tensor<T>& t2);

    template <typename U>
    Tensor<U> to() const;

    void print() const;
};

// Declare explicit instantiations (extern templates) for specific types
extern template class Tensor<float>;
extern template class Tensor<int32_t>;
extern template class Tensor<uint8_t>;

// Extern template instantiations for the `to<U>()` function
extern template Tensor<int32_t> Tensor<float>::to<int32_t>() const;
extern template Tensor<uint8_t> Tensor<float>::to<uint8_t>() const;
extern template Tensor<float> Tensor<int32_t>::to<float>() const;
extern template Tensor<uint8_t> Tensor<int32_t>::to<uint8_t>() const;
extern template Tensor<float> Tensor<uint8_t>::to<float>() const;
extern template Tensor<int32_t> Tensor<uint8_t>::to<int32_t>() const;

#endif // TENSOR_HPP
