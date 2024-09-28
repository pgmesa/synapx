
#include "tensor.hpp"
#include <iostream>
#include <exception>

int main() {
    try {
        using dtype = float32; 
        
        // Step 1: Create a tensor t1 of shape {2}, filled with 3.0 + 2.0
        Tensor<dtype> t1 = Tensor<dtype>::full({2}, 3.0) + 1.0;
        t1 += 1.0;
        t1 *= 2;
        std::cout << "Created tensor t1 with shape {2}, filled with 3.0:\n" << t1.to_string() << std::endl;

        // Step 2: Create a tensor t2 of shape {3, 2}, filled with 4.0
        Tensor<dtype> t2 = Tensor<dtype>::full({3, 2}, 4.0) * t1;
        std::cout << "Created tensor t2 with shape {3, 2}, filled with 4.0:\n" << t2.to_string() << std::endl;

        // Step 3: Expand t1 to shape {3, 2}
        t1 = t1.expand({3, 2});
        std::cout << "Expanded tensor t1 to shape {3, 2}:\n" << t1.to_string() << std::endl;

        // Step 4: Perform element-wise addition t3 = t1 + t2
        Tensor<dtype> t3 = (t1 + t2);
        std::cout << "Performed t1 + t2, resulting in tensor t3:\n" << t3.to_string() << std::endl;

        // Step 5: Reshape t3 to shape {2, -1}, which should infer the second dimension
        t3 = t3.view({2, -1});
        std::cout << "Reshaped tensor t3 to shape {2, -1} (second dimension inferred):\n" << t3.to_string() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[!] Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}