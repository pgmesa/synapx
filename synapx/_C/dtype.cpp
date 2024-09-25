
#include "dtype.hpp"


// Get the size of the data type
size_t get_dtype_size(const DataType dtype) {
    switch (dtype) {
        case DataType::UINT8:   return sizeof(uint8);
        case DataType::INT32:   return sizeof(int32);
        case DataType::FLOAT32: return sizeof(float32);
        default:                return 0;
    }
}

// Convert DataType to string
std::string dtype_to_str(const DataType dtype) {
    switch (dtype) {
        case DataType::UINT8:   return "uint8";
        case DataType::INT32:   return "int32";
        case DataType::FLOAT32: return "float32";
        default:                return "unknown";
    }
}