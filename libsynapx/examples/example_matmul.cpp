#include <chrono>
#include <exception>
#include <memory>

#include <synapx/synapx.hpp>
#include <argparse/argparse.hpp>
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

std::string shape_to_string(c10::IntArrayRef sizes) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < sizes.size(); ++i) {
        oss << sizes[i];
        if (i != sizes.size() - 1)
            oss << ", ";
    }
    oss << "]";
    return oss.str();
}

int main(int argc, char** argv) {
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

    // Setup argparse
    argparse::ArgumentParser program("synapx_test");

    program.add_argument("--t1-shape")
        .help("Shape of tensor t1 as comma-separated dimensions, e.g. 2,2,512,512")
        .default_value(std::string("2,2,512,512"));

    program.add_argument("--t2-shape")
        .help("Shape of tensor t2 as comma-separated dimensions, e.g. 512,512")
        .default_value(std::string("512,512"));

    try {
        program.parse_args(argc, argv);

        std::string t1_shape_str = program.get<std::string>("--t1-shape");
        std::string t2_shape_str = program.get<std::string>("--t2-shape");

        // Parse shapes
        std::vector<int64_t> t1_shape = parse_shape(t1_shape_str);
        std::vector<int64_t> t2_shape = parse_shape(t2_shape_str);

        spdlog::info("Starting Program");
        spdlog::info("t1 shape: {}", t1_shape_str);
        spdlog::info("t2 shape: {}", t2_shape_str);

        // Create two tensors
        torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);
        synapx::Tensor t1 = synapx::rand(t1_shape, true, options);
        synapx::Tensor t2 = synapx::rand(t2_shape, true, options);

        // Measure forward pass time
        auto start_forward = std::chrono::high_resolution_clock::now();

        spdlog::info("Forward");
        synapx::Tensor out = t1.matmul(t2);

        auto end_forward = std::chrono::high_resolution_clock::now();
        auto forward_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_forward - start_forward
        );

        // Measure backward pass time
        auto start_backward = std::chrono::high_resolution_clock::now();

        spdlog::info("Backward");
        out.backward(synapx::ones_like(out));

        auto end_backward = std::chrono::high_resolution_clock::now();
        auto backward_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_backward - start_backward
        );

        if (t1.grad().defined()) {
            spdlog::info("T1 has gradient with shape {}", shape_to_string(t1.grad().sizes()));
        }
        if (t2.grad().defined()) {
            spdlog::info("T2 has gradient with shape {}", shape_to_string(t2.grad().sizes()));
        }
        spdlog::info("Forward Time: {} ms", forward_duration.count());
        spdlog::info("Backward Time: {} ms", backward_duration.count());

    } catch (const std::exception& e) {
        spdlog::error("Exception caught: {}", e.what());
    } catch (...) {
        spdlog::error("An unknown exception occurred!");
    }

    return 0;
}