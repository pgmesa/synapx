
#include "utils.hpp"


std::vector<int> utils::calc_strides(const std::vector<int>& shape) {
    std::vector<int> strides(shape.size());
    size_t numel = 1;
    int i = static_cast<int>(shape.size()-1);
    for (i; i >= 0; i--) {
        strides[i] = static_cast<int>(numel);
        numel *= shape[i];
    }
    return strides;
}