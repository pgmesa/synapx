
#include <chrono>
#include <iostream>
#include <torch/torch.h> // Include PyTorch


// Function to measure time for matrix multiplication using torch::matmul
void torch_matmul() {
    auto A = torch::rand({2, 2, 512, 512}, torch::kFloat);
    auto B = torch::rand({512, 512}, torch::kFloat);

    // Print shapes of the input tensors
    std::cout << "Shape of A: ";
    for (const auto& dim : A.sizes()) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    std::cout << "Shape of B: ";
    for (const auto& dim : B.sizes()) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    auto C = torch::matmul(A, B);
    auto end = std::chrono::high_resolution_clock::now();

    // Print shape of the output tensor
    std::cout << "Shape of C: ";
    for (const auto& dim : C.sizes()) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "torch::matmul time: " << elapsed.count() * 1000 << " ms" << std::endl;
}

int main() {
    std::cout << "Running torch::matmul:" << std::endl;
    torch_matmul();

    return 0;
}