#include <iostream>
#include <chrono>
#include <stdexcept>

#include <synapx/tensor.hpp>


int main() {

    try {
        // Create two tensors
        synapx::Tensor t1(torch::rand({3, 3}));
        synapx::Tensor t2(torch::rand({3, 3}), true);

        // Measure forward pass time
        auto start_forward = std::chrono::high_resolution_clock::now();

        // Perform forward operation
        synapx::Tensor out = t1.add(t2);

        auto end_forward = std::chrono::high_resolution_clock::now();
        auto forward_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_forward - start_forward);

        // Measure backward pass time
        auto start_backward = std::chrono::high_resolution_clock::now();

        // Perform backward operation
        out.backward();

        auto end_backward = std::chrono::high_resolution_clock::now();
        auto backward_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_backward - start_backward);

        // Print results
        std::cout << "Forward Result:\n" << out.to_string() << "\n";
        std::cout << "Gradient for t1:\n" << t1.grad().value() << "\n";
        std::cout << "Gradient for t2:\n" << t2.grad().value() << "\n";
        std::cout << "Forward Time: " << forward_duration.count() << " microseconds\n";
        std::cout << "Backward Time: " << backward_duration.count() << " microseconds\n";

    } catch (const std::exception& e) {
        // Catch and print standard exceptions
        std::cerr << "Exception caught: " << e.what() << std::endl;
    } catch (...) {
        // Catch any other types of exceptions
        std::cerr << "An unknown exception occurred!" << std::endl;
    }

    return 0;
}
