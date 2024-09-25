
#include "utils.hpp"


std::vector<int> calc_strides(const std::vector<int>& shape, size_t dtype_size) {
    std::vector<int> strides(shape.size());
    size_t numel = 1;
    int i = static_cast<int>(shape.size()-1);
    for (i; i >= 0; i--) {
        strides[i] = static_cast<int>(numel * dtype_size); // NumPy strides 
        numel *= shape[i];
    }
    return strides;
}