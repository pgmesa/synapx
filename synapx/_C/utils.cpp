
#include "utils.hpp"
#include <utility>
#include <stdexcept>
#include <string>


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

bool utils::shapes_equal(const std::vector<int>& shape1, const std::vector<int>& shape2) {
    if (shape1.size() != shape2.size()) return false;
    for (int i=0; i < shape1.size(); i++) {
        if (shape1[i] != shape2[i]) return false;
    }
    return true;
}

std::vector<int> utils::broadcast_shapes(const std::vector<int>& shape1, const std::vector<int>& shape2) {
    int max_dims = std::max(shape1.size(), shape2.size());
    std::vector<int> result(max_dims);

    // Iterate from right to left (least significant dimension to most significant)
    for (int i = 1; i <= max_dims; ++i) {
        int dim1 = i <= shape1.size() ? shape1[shape1.size() - i] : 1;
        int dim2 = i <= shape2.size() ? shape2[shape2.size() - i] : 1;

        if (dim1 == dim2) {
            result[max_dims - i] = dim1;
        } else if (dim1 == 1) {
            result[max_dims - i] = dim2;
        } else if (dim2 == 1) {
            result[max_dims - i] = dim1;
        } else {
            throw std::runtime_error("Incompatible shapes for broadcasting: " + 
                                     std::to_string(dim1) + " and " + std::to_string(dim2) + 
                                     " at dimension " + std::to_string(max_dims - i));
        }
    }

    return result;
}

std::pair<std::vector<int>, std::vector<int>> utils::broadcast_shapes_for_matmul(
                const std::vector<int>& shape1, const std::vector<int>& shape2) {
    // Ensure shapes have at least 2 dimensions
    auto padded1 = shape1.size() < 2 ? std::vector<int>{1, shape1.back()} : shape1;
    auto padded2 = shape2.size() < 2 ? std::vector<int>{shape2.front(), 1} : shape2;

    // Check if the matrix dimensions are compatible
    if (padded1.back() != padded2[padded2.size() - 2]) {
        throw std::runtime_error("Incompatible dimensions for matrix multiplication: " + 
                                 std::to_string(padded1.back()) + " and " + 
                                 std::to_string(padded2[padded2.size() - 2]));
    }

    // Extract batch dimensions
    std::vector<int> batch1(padded1.begin(), padded1.end() - 2);
    std::vector<int> batch2(padded2.begin(), padded2.end() - 2);

    // Broadcast batch dimensions
    std::vector<int> broadcasted_batch;
    try {
        broadcasted_batch = broadcast_shapes(batch1, batch2);
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Error in broadcasting batch dimensions: ") + e.what());
    }

    // Construct the final shapes
    std::vector<int> result1 = broadcasted_batch;
    result1.insert(result1.end(), padded1.end() - 2, padded1.end());

    std::vector<int> result2 = broadcasted_batch;
    result2.insert(result2.end(), padded2.end() - 2, padded2.end());

    return {result1, result2};
}