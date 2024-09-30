#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "dtype.hpp"

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
    bool is_view = false;

    // Constructors
    Tensor();
    Tensor(const std::shared_ptr<T[]>& data, const std::vector<int>& shape);
    Tensor(const std::shared_ptr<T[]>& data, const std::vector<int>& shape, const std::vector<int>& strides);

    T get(int idx) const;
    T* get_ptr(int idx) const;

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

    static Tensor matmul(const Tensor& t1, const Tensor& t2);
    Tensor matmul(const Tensor& t2) const;
    
    // Tensor operator[](int index) const;

    Tensor view(const std::vector<int>& shape) const;
    Tensor expand(const std::vector<int>& shape) const;       // Broadcast
    Tensor broadcast_to(const std::vector<int>& shape) const; // Same as expand
    Tensor squeeze(const std::vector<int>& dims) const;
    Tensor unsqueeze(const std::vector<int>& dims) const;

    // String representation
    std::string to_string() const;
};

// Explicit instantiation declarations
extern template class Tensor<uint8>;
extern template class Tensor<int32>;
extern template class Tensor<float32>;

#endif