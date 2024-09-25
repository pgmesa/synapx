#ifndef DTYPE_HPP
#define DTYPE_HPP

#include <cstdint>
#include <iostream>


// Define aliases for types
using uint8 = uint8_t;
using int32 = int32_t;
using float32 = float;

// Enum class for data types
enum class DataType {
    UINT8,
    INT32,
    FLOAT32,
};


size_t get_dtype_size(const DataType dtype);
std::string dtype_to_str(const DataType dtype);

// Function to get DataType from C++ type
template<typename T>
DataType get_dtype() {
    if (std::is_same<T, uint8>::value) {
        return DataType::UINT8;
    } else if (std::is_same<T, int32>::value) {
        return DataType::INT32;
    } else if (std::is_same<T, float32>::value) {
        return DataType::FLOAT32;
    } 
    throw std::runtime_error("Unsupported type for Tensor");
}

#endif