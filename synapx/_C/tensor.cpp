
#include "tensor.hpp"
#include "functional.hpp"


template class Tensor<uint8>;
template class Tensor<int32>;
template class Tensor<float32>;

// ================================= Initializers =================================
template<typename T>
Tensor<T> Tensor<T>::empty(const std::vector<int>& shape) {
    int numel = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    std::shared_ptr<T[]> data = std::shared_ptr<T[]>(new T[numel]);
    return Tensor<T>(data, numel, shape);
}

template<typename T>
Tensor<T> Tensor<T>::full(const std::vector<int>& shape, double value) {
    Tensor<T> tensor = Tensor<T>::empty(shape);
    T cvalue = cast_value<T>(value);

    for (int i = 0; i < tensor.numel; i++) {
        tensor.data[i] = cvalue;
    }
    return tensor;
}

template<typename T>
Tensor<T> Tensor<T>::ones(const std::vector<int>& shape) {
    return Tensor<T>::full(shape, 1.0);
}

template<typename T>
Tensor<T> Tensor<T>::zeros(const std::vector<int>& shape) {
    return Tensor<T>::full(shape, 0.0);
}
// ================================================================================

// Default constructor
template<typename T>
Tensor<T>::Tensor()
    : data(nullptr), numel(0), shape{}, ndim(0), dtype(DataType::FLOAT32), strides{} {}

template<typename T>
Tensor<T>::Tensor(
    const std::shared_ptr<T[]>& data,
    const size_t numel,
    const std::vector<int>& shape) 
    : data(data), numel(numel), shape(shape),
      ndim(static_cast<int>(shape.size())), dtype(get_dtype<T>()), strides(calc_strides(shape, sizeof(T))) {

    size_t numel_check = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    if (numel != numel_check) {
        throw std::runtime_error("numel does not match the product of shape elements.");
    }
}

template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& t2) const {
    return F::add(*this, t2);
}

// template<typename T>
// Tensor<T> Tensor<T>::operator+(const double value) const {
//     return F::add(*this, t2);
// }

// template<typename T>
// Tensor<T>& Tensor<T>::operator+=(const double value) {
//     return F::add(*this, value);
// }

// Tensor Tensor::operator+(const double value) const {
//     Tensor out = std::visit([this, value](auto& data_ptr) -> Tensor {
//         using T = std::decay_t<decltype(data_ptr.get()[0])>;
//         return this->element_wise_operation_scalar<T>(value, std::plus<T>());   
//     }, this->data);
//     return out;
// }

// Tensor& Tensor::operator+=(const Tensor& t2) {
//     std::visit([this, &t2](auto& data_ptr) {
//         using T = std::decay_t<decltype(data_ptr.get()[0])>;
//         this->element_wise_operation_in_place<T>(t2, std::plus<T>());
//     }, this->data);

//     return *this;
// }

// Tensor& Tensor::operator+=(const double value) {
//     std::visit([this, value](auto& data_ptr) {
//         using T = std::decay_t<decltype(data_ptr.get()[0])>; 
//         this->element_wise_operation_in_place_scalar<T>(value, std::plus<T>());
//     }, this->data);

//     return *this;
// }

// Tensor Tensor::operator*(const Tensor& t2) const {
//     Tensor out = std::visit([this, &t2](auto& data_ptr) -> Tensor {
//         using T = std::decay_t<decltype(data_ptr.get()[0])>;
//         return this->element_wise_operation<T>(t2, std::multiplies<T>());
//     }, this->data);

//     return out;
// }

// Tensor Tensor::operator*(const double value) const {
//     Tensor out = std::visit([this, value](auto& data_ptr) -> Tensor {
//         using T = std::decay_t<decltype(data_ptr.get()[0])>;
//         return this->element_wise_operation_scalar<T>(value, std::multiplies<T>());   
//     }, this->data);
//     return out;
// }

// Tensor& Tensor::operator*=(const Tensor& t2) {
//     std::visit([this, &t2](auto& data_ptr) {
//         using T = std::decay_t<decltype(data_ptr.get()[0])>;
//         this->element_wise_operation_in_place<T>(t2, std::multiplies<T>());
//     }, this->data);

//     return *this;
// }

// Tensor& Tensor::operator*=(const double value) {
//     std::visit([this, value](auto& data_ptr) {
//         using T = std::decay_t<decltype(data_ptr.get()[0])>; 
//         this->element_wise_operation_in_place_scalar<T>(value, std::multiplies<T>());
//     }, this->data);

//     return *this;
// }

// Tensor Tensor::operator[](int index) const {
//     if (index < 0 || index >= this->shape[0]) {
//         throw std::out_of_range("Index out of range");
//     }

//     DataType out_dtype = this->dtype;
//     size_t out_numel = this->numel / this->shape[0];
    
//     std::vector<int> out_shape(this->shape.begin() + 1, this->shape.end());

//     size_t offset = index * out_numel;

//     TensorData out_data = std::visit([offset](auto& data_ptr) -> TensorData {
//         using T = std::decay_t<decltype(data_ptr.get()[0])>;
//         // Create a new shared_ptr that points to the correct offset
//         return std::shared_ptr<T[]>(data_ptr, data_ptr.get() + offset);
//     }, this->data);

//     return Tensor(out_data, out_numel, out_shape, out_dtype);
// }


template<typename T>
std::string Tensor<T>::to_string() const {
    size_t mem_size = this->numel * get_dtype_size(this->dtype);
    std::string dtype_str = dtype_to_str(this->dtype);

    std::string prefix = "Tensor(";
    int padding = prefix.size(); 
    std::string spaces(padding, ' ');

    std::string data_str = ndarray_to_string(this->data.get(), this->numel, this->shape, padding);
    std::string shape_str = array_to_string(shape.data(), shape.size(), 0);
    std::string strides_str = array_to_string(strides.data(), strides.size(), 0);
    
    std::ostringstream oss;
    oss << prefix << data_str << ",\n"
        << spaces << "numel=" << numel << ", shape=" << shape_str << ", ndim=" << ndim
        << ", strides=" << strides_str << ", dtype=" << dtype_str << ", msize=" << mem_size << ")";
    return oss.str();
}