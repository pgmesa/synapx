
#include "tensor.hpp"
#include "utils.hpp"
#include "functional.hpp"

#include <stdexcept>
#include <numeric>
#include <algorithm>

template class Tensor<uint8>;
template class Tensor<int32>;
template class Tensor<float32>;


// Default constructor
template<typename T>
Tensor<T>::Tensor()
    : data(nullptr), numel(0), shape{}, ndim(0), dtype(DataType::FLOAT32), strides{} {}

template<typename T>
Tensor<T>::Tensor(
    const std::shared_ptr<T[]>& data,
    const std::vector<int>& shape) 
    : 
    data(data), 
    numel(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>())), 
    shape(shape),
    ndim(static_cast<int>(shape.size())),
    dtype(get_dtype<T>()),
    strides(utils::calc_strides(shape)) {}

template<typename T>
Tensor<T>::Tensor(
    const std::shared_ptr<T[]>& data,
    const std::vector<int>& shape,
    const std::vector<int>& strides)
    :
    data(data),
    numel(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>())),
    shape(shape),
    ndim(static_cast<int>(shape.size())),
    dtype(get_dtype<T>()),
    strides(strides) {}


template<typename T>
Tensor<T> Tensor<T>::empty(const std::vector<int>& shape) {
    int numel = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    std::shared_ptr<T[]> data = std::shared_ptr<T[]>(new T[numel]);
    return Tensor<T>(data, shape);
}

template<typename T>
Tensor<T> Tensor<T>::full(const std::vector<int>& shape, double value) {
    Tensor<T> tensor = Tensor<T>::empty(shape);
    T cvalue = utils::cast_value<T>(value);

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


template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& t2) const {
    return F::add(*this, t2);
}

template<typename T>
Tensor<T>& Tensor<T>::operator+=(const Tensor<T>& t2) {
    F::add(*this, t2, true);
    return *this;
}

template<typename T>
Tensor<T> Tensor<T>::operator+(const double value) const {
    Tensor<T> t2 = Tensor<T>::full({1}, value);
    return F::add(*this, t2);
}

template<typename T>
Tensor<T>& Tensor<T>::operator+=(const double value) {
    Tensor<T> t2 = Tensor<T>::full({1}, value);
    F::add(*this, t2, true);
    return *this;
}


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


template<typename T>
T Tensor<T>::get(int idx) const {
    return *(this->get_ptr(idx));
}

template<typename T>
T* Tensor<T>::get_ptr(int idx) const {
    int offset = 0;
    for (int i=0; i < this->ndim; i++) {
        if (idx < this->strides[i]) continue;
        int blocks = idx % this->shape[i];
        int nelem = blocks * this->strides[i];
        offset += nelem;
        idx -= nelem;
    }
    return this->data.get() + offset;
}

template<typename T>
Tensor<T> Tensor<T>::view(const std::vector<int>& shape) const {
    std::vector<int> new_shape(shape);
    // Check if there's at most one -1 in the new shape
    int neg_one_count = std::count(new_shape.begin(), new_shape.end(), -1);
    if (neg_one_count > 1) {
        throw std::invalid_argument("Only one -1 is allowed in the new shape");
    }

    // Calculate the total number of elements in the new shape
    int total_elements = 1;
    int neg_one_index = -1;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            neg_one_index = i;
        } else if (new_shape[i] > 0) {
            total_elements *= new_shape[i];
        } else {
            throw std::invalid_argument("Invalid dimension size");
        }
    }

    // If there's a -1, calculate its value
    if (neg_one_index != -1) {
        if (numel % total_elements != 0) {
            throw std::invalid_argument("Cannot reshape tensor to requested dimensions");
        }
        new_shape[neg_one_index] = numel / total_elements;
        total_elements *= new_shape[neg_one_index];
    }

    // Check if the total number of elements matches
    if (total_elements != numel) {
        throw std::invalid_argument("New shape does not match the number of elements in the tensor");
    }

    // Create a new Tensor with the same data but new shape
    Tensor<T> viewed_tensor(this->data, new_shape);
    viewed_tensor.is_view = true;

    return viewed_tensor;
}

template<typename T>
Tensor<T> Tensor<T>::expand(const std::vector<int>& shape) const {
    // Check if the new shape is compatible for broadcasting
    if (shape.size() < this->shape.size()) {
        throw std::invalid_argument("New shape must have at least as many dimensions as the current shape");
    }

    std::vector<int> current_shape = this->shape;
    std::vector<int> current_strides = this->strides;

    // Pad the current shape and strides with 1's and 0's at the front if necessary
    while (current_shape.size() < shape.size()) {
        current_shape.insert(current_shape.begin(), 1);
        current_strides.insert(current_strides.begin(), 0);
    }

    std::vector<int> new_strides(shape.size());

    // Check compatibility and compute new strides
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] < current_shape[i]) {
            std::ostringstream oss;
            oss << "Cannot broadcast to a smaller size. Dimension " << i 
                << " of input is " << current_shape[i] 
                << ", but dimension " << i << " of new shape is " << shape[i];
            throw std::invalid_argument(oss.str());
        }
        if (current_shape[i] == shape[i]) {
            new_strides[i] = current_strides[i];
        } else if (current_shape[i] == 1) {
            new_strides[i] = 0;  // Broadcast this dimension
        } else {
            std::ostringstream oss;
            oss << "Shape is not compatible for broadcasting at dimension " << i 
                << ". Input shape is " << current_shape[i] 
                << ", but new shape is " << shape[i] 
                << ". For broadcasting, dimensions must either be equal or input dimension must be 1.";
            throw std::invalid_argument(oss.str());
        }
    }

    // Create a new tensor with the same data but new shape and strides
    Tensor<T> expanded_tensor(this->data, shape, new_strides);
    expanded_tensor.is_view = true;

    return expanded_tensor;
}


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

    std::string data_str = utils::tensor_to_string(*this, padding);
    std::string shape_str = utils::array_to_string(shape.data(), shape.size(), 0);
    std::string strides_str = utils::array_to_string(strides.data(), strides.size(), 0);
    
    std::ostringstream oss;
    oss << prefix << data_str << ",\n"
        << spaces << "numel=" << numel << ", shape=" << shape_str << ", ndim=" << ndim
        << ", strides=" << strides_str << ", dtype=" << dtype_str << ", msize=" << mem_size << ")";
    return oss.str();
}