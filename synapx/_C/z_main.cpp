
#include "tensor.hpp"

int main() {
    using dtype = uint8; 
    Tensor<dtype> t1 = Tensor<dtype>::full({1,2,3}, 3.0);
    Tensor<dtype> t2 = Tensor<dtype>::full({1,2,3}, 3.0);
     
    std::cout << (t1 + t2).to_string();
}