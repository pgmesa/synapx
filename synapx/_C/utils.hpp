#ifndef UTILS_HPP
#define UTILS_HPP

#include <xtensor/xtensor.hpp>
#include <stdexcept>          
#include <utility>             
#include <iostream>
#include <sstream>
#include <numeric>
#include <iomanip>
#include <limits>


namespace utils {

    const uint8_t DECIMALS = 4;
    const uint8_t VALUES_PER_LINE = 8;

    /**
     * @brief Broadcast two shapes to a common shape.
     * 
     * This function performs broadcasting between two shapes, similar to how broadcasting
     * works in NumPy and xtensor. If the shapes are incompatible for broadcasting,
     * it throws an exception.
     * 
     * @param shape1 The first shape as an std::vector<size_t>.
     * @param shape2 The second shape as an std::vector<size_t>.
     * @return The broadcasted shape as an std::vector<size_t>.
     * @throws std::runtime_error if the shapes are incompatible for broadcasting.
     */
    std::vector<size_t> broadcast_shapes(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2);

    /**
     * @brief Broadcast shapes for matrix multiplication.
     * 
     * This function ensures that the given shapes are compatible for matrix multiplication.
     * It pads the shapes to ensure they have at least two dimensions and checks that
     * the inner dimensions are compatible. It also broadcasts the batch dimensions
     * (if present) and returns the broadcasted shapes for the two matrices.
     * 
     * @param shape1 The first shape as an std::vector<size_t>.
     * @param shape2 The second shape as an std::vector<size_t>.
     * @return A pair of broadcasted shapes for the two matrices as a std::pair<std::vector<size_t>, std::vector<size_t>>.
     * @throws std::runtime_error if the shapes are incompatible for matrix multiplication or broadcasting.
     */
    std::pair<std::vector<size_t>, std::vector<size_t>> broadcast_shapes_for_matmul(
        const std::vector<size_t>& shape1, const std::vector<size_t>& shape2
    );

    template <typename T>
    std::string array_to_string(const T* array, size_t length, int padding=0) {
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

    template <typename T>
    std::string vector_to_string(const std::vector<T>& vector, int padding=0) {
        return array_to_string(vector.data(), vector.size(), padding);
    }

} // namespace utils

#endif // UTILS_HPP
