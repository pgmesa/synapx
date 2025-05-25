#include <chrono>
#include <exception>
#include <memory>

#include <synapx/tensor.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/pattern_formatter.h>


int main() {
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

    try {
        spdlog::info("Starting Program");

        // Create two tensors
        synapx::Tensor t1(torch::rand({3, 3}), false);
        synapx::Tensor t2(torch::rand({3, 3}), true);

        // Measure forward pass time
        auto start_forward = std::chrono::high_resolution_clock::now();

        spdlog::info("Forward");
        synapx::Tensor out = t1.add(t2);

        auto end_forward = std::chrono::high_resolution_clock::now();
        auto forward_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_forward - start_forward
        );

        // Measure backward pass time
        auto start_backward = std::chrono::high_resolution_clock::now();

        spdlog::info("Backward");
        out.backward();

        auto end_backward = std::chrono::high_resolution_clock::now();
        auto backward_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_backward - start_backward
        );

        // Print results
        if (t1.grad().defined()) {
            spdlog::info("Gradient for t1:\n{}", synapx::Tensor::to_string(t1.grad()));
        }
        if (t2.grad().defined()) {
            spdlog::info("Gradient for t2:\n{}",  synapx::Tensor::to_string(t2.grad()));
        }
        spdlog::info("Forward Result:\n{}", out.to_string());
        spdlog::info("Forward Time: {} ms", forward_duration.count());
        spdlog::info("Backward Time: {} ms", backward_duration.count());

    } catch (const std::exception& e) {
        spdlog::error("Exception caught: {}", e.what());
    } catch (...) {
        spdlog::error("An unknown exception occurred!");
    }

    return 0;
}