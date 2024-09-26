#ifndef UTILS_HPP
#define UTILS_HPP

#include "tensor.hpp"

#include <vector>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <numeric>
#include <iomanip>
#include <limits>

const int DECIMALS = 4;
const int VALUES_PER_LINE = 8;

namespace utils {

std::vector<int> calc_strides(const std::vector<int>& shape);

template<typename T>
T cast_value(double value) {
    T tvalue;

    if (value > std::numeric_limits<T>::max()) {
        tvalue = std::numeric_limits<T>::max();
    } else if (value < std::numeric_limits<T>::min()) {
        tvalue = std::numeric_limits<T>::min();
    } else {
        tvalue = static_cast<T>(value);
    }
    
    return tvalue;
}

template <typename T>
std::string array_to_string(const T* array, size_t length, int padding) {
    std::ostringstream oss;
    oss << "[";

    for (int i = 0; i < length; ++i) {
        // Check if T is an integral type (e.g., int, uint8_t)
        if constexpr (std::is_integral_v<T>) {
            oss << static_cast<int>(array[i]);
        } 
        // Check if T is a floating-point type (e.g., float, double)
        else if constexpr (std::is_floating_point_v<T>) {
            oss << std::fixed << std::setprecision(DECIMALS) << array[i];
        } 
        else {
            throw std::invalid_argument("Unsupported data type.");
        }

        if (i < length - 1) {
            oss << ", ";
            if ((i + 1) % VALUES_PER_LINE == 0) {
                oss << "\n";
                // Add padding spaces
                oss << std::string(padding, ' ');
            }
        }
    }
    oss << "]";

    return oss.str();
}

template<typename T>
std::string tensor_to_string(const Tensor<T>& tensor, int padding) {
    int ndim = tensor.ndim;
    int last_dim_size = tensor.shape[ndim - 1];
    size_t narrays = tensor.numel / last_dim_size;
    std::vector<int> strides = calc_strides(tensor.shape);

    std::string buffer;
    int dims_ended = (ndim - 1);

    for (int i = 0; i < narrays; i++) {
        if (i > 0) {
            int spaces = padding + (ndim - 1 - dims_ended);
            buffer.append(spaces, ' ');
        }
        buffer.append(dims_ended, '[');

        // Get array string
        T* array = tensor.get_ptr(i*last_dim_size);
        std::string array_str = array_to_string(array, last_dim_size, 7);
        buffer.append(array_str);

        // Calculate finished matrices
        dims_ended = 0;
        for (int sidx = 0; sidx < ndim - 1; sidx++) {
            if (sidx == ndim - 2) continue;
            if (((i + 1) * last_dim_size) % strides[sidx] == 0) {
                dims_ended += 1;
            }
        }
        buffer.append(dims_ended, ']');

        // Not the last array
        if (i < (narrays - 1)) {
            buffer.append(",\n");
            buffer.append(dims_ended, '\n');
        }
    }
    if (ndim > 1) buffer.append("]");
    return buffer;
}

}

#endif