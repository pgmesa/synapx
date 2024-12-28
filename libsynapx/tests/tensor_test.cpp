
#include <torch/torch.h>
#include <synapx/tensor.hpp>
#include <iostream>


int main() {
    // Test basic torch operations
    torch::Tensor a = torch::randn({2, 3});
    torch::Tensor b = torch::randn({3, 2});
    
    std::cout << "Tensor A:" << std::endl;
    std::cout << a << std::endl;
    std::cout << "Tensor B:" << std::endl;
    std::cout << b << std::endl;
    
    // Test matrix multiplication with torch
    torch::Tensor c = torch::matmul(a, b);
    std::cout << "Matrix multiplication result (torch):" << std::endl;
    std::cout << c << std::endl;
    
    // Test our custom Tensor wrapper
    synapx::Tensor custom_tensor(a);
    std::cout << "\nTesting our custom Tensor wrapper:" << std::endl;
    std::cout << "Number of dimensions: " << custom_tensor.ndim() << std::endl;
    std::cout << "Number of elements: " << custom_tensor.numel() << std::endl;
    
    // Verify dimensions
    if (custom_tensor.ndim() != 2) {
        std::cerr << "Error: Expected 2 dimensions, got " << custom_tensor.ndim() << std::endl;
        return 1;
    }
    
    if (custom_tensor.numel() != 6) {  // 2x3 tensor should have 6 elements
        std::cerr << "Error: Expected 6 elements, got " << custom_tensor.numel() << std::endl;
        return 1;
    }
    
    std::cout << "\nAll tests passed successfully!" << std::endl;
    return 0;
}