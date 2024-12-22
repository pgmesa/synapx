
#include "utils.hpp"
#include <vector> 
#include <stdexcept>
#include <string>


std::vector<size_t> utils::broadcast_shapes(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2) {
    size_t max_dims = std::max(shape1.size(), shape2.size());
    std::vector<size_t> result(max_dims);

    // Iterate from right to left (least significant dimension to most significant)
    for (size_t i = 1; i <= max_dims; ++i) {
        size_t dim1 = i <= shape1.size() ? shape1[shape1.size() - i] : 1;
        size_t dim2 = i <= shape2.size() ? shape2[shape2.size() - i] : 1;

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

std::pair<std::vector<size_t>, std::vector<size_t>> utils::broadcast_shapes_for_matmul(
    const std::vector<size_t>& shape1, const std::vector<size_t>& shape2) {

    // Ensure shapes have at least 2 dimensions
    std::vector<size_t> padded1 = shape1.size() < 2 ? std::vector<size_t>{1, shape1.back()} : shape1;
    std::vector<size_t> padded2 = shape2.size() < 2 ? std::vector<size_t>{shape2.front(), 1} : shape2;

    // Check if the matrix dimensions are compatible
    if (padded1.back() != padded2[padded2.size() - 2]) {
        throw std::runtime_error("Incompatible dimensions for matrix multiplication: " + 
                                 std::to_string(padded1.back()) + " and " + 
                                 std::to_string(padded2[padded2.size() - 2]));
    }

    // Extract batch dimensions
    std::vector<size_t> batch1(padded1.begin(), padded1.end() - 2);
    std::vector<size_t> batch2(padded2.begin(), padded2.end() - 2);

    // Broadcast batch dimensions
    std::vector<size_t> broadcasted_batch;
    try {
        broadcasted_batch = broadcast_shapes(batch1, batch2);
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Error in broadcasting batch dimensions: ") + e.what());
    }

    // Construct the final shapes
    std::vector<size_t> result1 = broadcasted_batch;
    result1.insert(result1.end(), padded1.end() - 2, padded1.end());

    std::vector<size_t> result2 = broadcasted_batch;
    result2.insert(result2.end(), padded2.end() - 2, padded2.end());

    return {result1, result2};
}