#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "dtype.hpp"
#include "utils.hpp"

#include <iostream>
#include <vector>
#include <memory>


template<typename T>
class Tensor;
// Type trait to check for allowed types
template<typename T>
struct is_allowed_tensor_type : std::false_type {};
template<> struct is_allowed_tensor_type<uint8> : std::true_type {};
template<> struct is_allowed_tensor_type<int32> : std::true_type {};
template<> struct is_allowed_tensor_type<float32> : std::true_type {};


template<typename T>
class Tensor {
    // Manually check if the provided type is valid. Otherwise, it will raise a linker error but less intuitive
    static_assert(is_allowed_tensor_type<T>::value, "Tensor only supports uint8, int32, and float32");
public:
    std::shared_ptr<T[]> data;
    size_t numel;
    std::vector<int> shape;
    int ndim;
    DataType dtype;
    std::vector<int> strides;

    // Constructors
    Tensor();
    Tensor(const std::shared_ptr<T[]>& data, const size_t numel, const std::vector<int>& shape);

    // Initializers
    static Tensor empty(const std::vector<int>& shape);
    static Tensor full(const std::vector<int>& shape, double value);
    static Tensor ones(const std::vector<int>& shape);
    static Tensor zeros(const std::vector<int>& shape);

    // Operator overloads
    Tensor operator+(const Tensor& t2) const;
    Tensor operator+(const double value) const;
    Tensor& operator+=(const Tensor& t2);
    Tensor& operator+=(const double value);
    Tensor operator*(const Tensor& t2) const;
    Tensor operator*(const double value) const;
    Tensor& operator*=(const Tensor& t2);
    Tensor& operator*=(const double value);
    
    Tensor operator[](int index) const;

    // TODO: Implement to 
    Tensor view();
    Tensor expand(std::vector<int> shape); // Broadcast

    // String representation
    std::string to_string() const;
};

// Explicit instantiation declarations
extern template class Tensor<uint8>;
extern template class Tensor<int32>;
extern template class Tensor<float32>;

#endif