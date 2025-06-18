#include <chrono>
#include <exception>
#include <memory>
#include <string>
#include <vector>
#include <sstream>

#include <synapx/synapx.hpp>
#include <argparse/argparse.hpp>
#include <fmt/core.h>
#include <spdlog/spdlog.h>
#include <spdlog/pattern_formatter.h>


// Helper: parse shape string like "2,2,3,3" → vector<int64_t>
std::vector<int64_t> parse_shape(const std::string& s) {
    std::vector<int64_t> shape;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try {
            shape.push_back(std::stoll(item));
        } catch (const std::exception& e) {
            throw std::invalid_argument("Invalid shape dimension: " + item);
        }
    }
    if (shape.empty())
        throw std::invalid_argument("Shape string must contain at least one dimension");
    return shape;
}

int main(int argc, char** argv) {
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

    // Setup argparse
    argparse::ArgumentParser program("synapx_test");

    program.add_argument("--t1-shape")
        .help("Shape of tensor t1 as comma-separated dimensions, e.g. 2,2,3,3")
        .default_value(std::string("2,2,3,3"));

    program.add_argument("--t2-shape")
        .help("Shape of tensor t2 as comma-separated dimensions, e.g. 3,3")
        .default_value(std::string("3,3"));

    try {
        program.parse_args(argc, argv);
        
        spdlog::info("Testing NoGradGuard");
        synapx::Tensor x = synapx::ones({3, 4}, true);
        {
            synapx::autograd::NoGradGuard guard;
            auto y = x.add(x);
            spdlog::info("Requires grad after operation? {}", y.requires_grad());
        }

        std::string t1_shape_str = program.get<std::string>("--t1-shape");
        std::string t2_shape_str = program.get<std::string>("--t2-shape");

        // Parse shapes
        std::vector<int64_t> t1_shape = parse_shape(t1_shape_str);
        std::vector<int64_t> t2_shape = parse_shape(t2_shape_str);

        spdlog::info("Starting Program");
        spdlog::info("t1 shape: {}", t1_shape_str);
        spdlog::info("t2 shape: {}", t2_shape_str);

        synapx::Tensor t1 = synapx::rand(t1_shape, true);
        synapx::Tensor t2 = synapx::rand(t2_shape, true);

        auto start_forward = std::chrono::high_resolution_clock::now();

        spdlog::info("Forward");
        synapx::Tensor out = t1 + t2;

        auto end_forward = std::chrono::high_resolution_clock::now();
        auto forward_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_forward - start_forward
        );

        auto start_backward = std::chrono::high_resolution_clock::now();

        spdlog::info("Backward");
        out.sum().backward();

        auto end_backward = std::chrono::high_resolution_clock::now();
        auto backward_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_backward - start_backward
        );

        if (t1.grad().defined()) {
            spdlog::info("Gradient for t1:\n{}", t1.grad().to_string());
        }
        if (t2.grad().defined()) {
            spdlog::info("Gradient for t2:\n{}", t2.grad().to_string());
        }
        spdlog::info("Forward Result:\n{}", out.to_string());
        spdlog::info("Forward Time: {} ms", forward_duration.count());
        spdlog::info("Backward Time: {} ms", backward_duration.count());
            
    } catch (const std::exception& e) {
        spdlog::error("Exception caught: {}", e.what());
        return 1;
    } catch (...) {
        spdlog::error("An unknown exception occurred!");
        return 1;
    }

    return 0;
}
