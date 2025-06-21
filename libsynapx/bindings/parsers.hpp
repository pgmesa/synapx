#ifndef BINDINGS_PARSERS_HPP
#define BINDINGS_PARSERS_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <torch/extension.h>
#include <torch/torch.h>

#include <synapx/tensor.hpp>


namespace py = pybind11;

#if defined(_MSC_VER) && !defined(ssize_t)
typedef std::ptrdiff_t ssize_t;
#endif


// Converts a numpy.ndarray (py::array) → torch::Tensor
torch::Tensor numpy_to_torch(py::array array) {
    // Make sure it’s contiguous:
    if (!(array.flags() & py::array::c_style)) {
        array = py::array::ensure(array, py::array::c_style);
    }

    // Raw data pointer and shape:
    void* data_ptr = array.mutable_data();
    std::vector<int64_t> shape(array.shape(), array.shape() + array.ndim());

    // Figure out dtype:
    torch::Dtype dtype;
    if (array.dtype().is(py::dtype::of<float>()))            { dtype = torch::kFloat32; }
    else if (array.dtype().is(py::dtype::of<double>()))      { dtype = torch::kFloat64; }
    else if (array.dtype().is(py::dtype::of<int32_t>()))     { dtype = torch::kInt32; }
    else if (array.dtype().is(py::dtype::of<int64_t>()))     { dtype = torch::kInt64; }
    else if (array.dtype().is(py::dtype::of<uint8_t>()))     { dtype = torch::kUInt8; }
    else if (array.dtype().is(py::dtype::of<int8_t>()))      { dtype = torch::kInt8; }
    else if (array.dtype().is(py::dtype::of<int16_t>()))     { dtype = torch::kInt16; }
    else {
        throw std::runtime_error("Unsupported NumPy data type for tensor conversion");
    }

    // Create a tensor from the raw buffer, then clone so PyTorch owns its own memory:
    auto tens = torch::from_blob(data_ptr, shape, dtype);
    return tens.clone();
}

// Converts a torch::Tensor → numpy.ndarray
py::array torch_to_numpy(const torch::Tensor& torch_tensor) {
    auto contig = torch_tensor.contiguous();
    void* data_ptr = contig.data_ptr();
    std::vector<ssize_t> shape(contig.sizes().begin(), contig.sizes().end());
    std::vector<ssize_t> strides(contig.strides().begin(), contig.strides().end());

    // Convert strides (in elements) → strides in bytes:
    for (auto& st : strides) {
        st *= contig.element_size();
    }

    // Figure out numpy dtype:
    py::dtype dtype;
    switch (contig.scalar_type()) {
        case torch::kFloat32: dtype = py::dtype::of<float>();   break;
        case torch::kFloat64: dtype = py::dtype::of<double>();  break;
        case torch::kInt32:   dtype = py::dtype::of<int32_t>(); break;
        case torch::kInt64:   dtype = py::dtype::of<int64_t>(); break;
        case torch::kUInt8:   dtype = py::dtype::of<uint8_t>(); break;
        case torch::kInt8:    dtype = py::dtype::of<int8_t>();  break;
        case torch::kInt16:   dtype = py::dtype::of<int16_t>(); break;
        default:
            throw std::runtime_error("Unsupported tensor data type for conversion to NumPy");
    }

    // Return an array that (non‐copying) wraps the PyTorch buffer:
    return py::array(
        dtype,
        shape,
        strides,
        data_ptr,
        // Attach a capsule so PyTorch’s memory isn’t freed prematurely:
        py::capsule(data_ptr, [](void* p){ /* no-op; PyTorch owns it */ })
    );
}


class TensorIndexConverter {
public:
    static std::vector<torch::indexing::TensorIndex> convert(const py::object& key, const synapx::Tensor& tensor) {
        std::vector<torch::indexing::TensorIndex> indices;
        auto shape = tensor.shape();
        
        if (py::isinstance<py::tuple>(key)) {
            py::tuple tuple_key = key.cast<py::tuple>();
            for (size_t i = 0; i < tuple_key.size() && i < shape.size(); ++i) {
                indices.push_back(convert_single(tuple_key[i], shape[i]));
            }
        } else {
            indices.push_back(convert_single(key, shape.empty() ? 1 : shape[0]));
        }
        
        return indices;
    }
    
private:
    static torch::indexing::TensorIndex convert_single(const py::object& item, int64_t dim_size) {
        if (py::isinstance<py::int_>(item)) {
            int64_t idx = item.cast<int64_t>();
            // Handle negative indexing
            if (idx < 0) idx += dim_size;
            return torch::indexing::TensorIndex(idx);
        }
        else if (py::isinstance<py::slice>(item)) {
            py::slice slice_obj = item.cast<py::slice>();
            
            // Handle Python slice semantics
            auto start = slice_obj.attr("start");
            auto stop = slice_obj.attr("stop");
            auto step = slice_obj.attr("step");
            
            int64_t start_val = start.is_none() ? 0 : start.cast<int64_t>();
            int64_t stop_val = stop.is_none() ? dim_size : stop.cast<int64_t>();
            int64_t step_val = step.is_none() ? 1 : step.cast<int64_t>();
            
            // Handle negative indices
            if (start_val < 0) start_val += dim_size;
            if (stop_val < 0) stop_val += dim_size;
            
            return torch::indexing::Slice(start_val, stop_val, step_val);
        }
        else if (item.is_none()) {
            // Handle ":" slice
            return torch::indexing::Slice();
        }
        else {
            throw std::runtime_error("Unsupported index type");
        }
    }
};


synapx::Reduction get_reduction(const std::string& reduction) {
    synapx::Reduction reduction_;

    if (reduction == "none") {
        reduction_ = synapx::Reduction::None;
    } else if (reduction == "mean") {
        reduction_ = synapx::Reduction::Mean;
    } else if (reduction == "sum") {
        reduction_ = synapx::Reduction::Sum;
    } else {
        throw std::invalid_argument(reduction + " is not a valid value for reduction");
    }

    return reduction_;
}

#endif