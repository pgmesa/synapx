#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <tuple>
#include <memory>
#include <vector>
#include <cstddef>
#include <optional>

#include <torch/torch.h>

#include <synapx/core.hpp>


namespace synapx {

    class Tensor;
    class TensorIterator;
    namespace autograd { class Node; }

    using IntArray = std::vector<int64_t>;
    using TensorList = std::vector<Tensor>;
    using TensorIndices = std::vector<torch::indexing::TensorIndex>;
    
    class SYNAPX_API Tensor {
    public:
        Tensor();
        Tensor(const torch::Tensor& data, bool requires_grad=false);

        torch::Tensor data() const;
        void set_data(const Tensor& other);

        bool defined() const;
        bool requires_grad() const;
        torch::Dtype dtype() const;
        torch::Device device() const;
        torch::TensorOptions options() const;

        size_t numel() const;
        size_t dim() const;
        IntArray shape() const;
        torch::IntArrayRef sizes() const;
        size_t size(int64_t dim) const;

        bool is_leaf() const;
        void retain_grad();
        bool retains_grad() const;
        bool is_floating_point() const;

        torch::Scalar item() const;
        Tensor detach() const;
        Tensor count_nonzero() const;
        Tensor argmax(std::optional<int64_t> dim = std::nullopt, bool keepdim = false) const;
        Tensor argmin(std::optional<int64_t> dim = std::nullopt, bool keepdim = false) const;

        const Tensor grad() const;
        void set_grad(const Tensor& grad);

        std::shared_ptr<autograd::Node> grad_fn() const;
        void set_grad_fn(std::shared_ptr<autograd::Node> grad_fn);

        void backward(const Tensor& grad={});

        Tensor operator+(const Tensor& other) const;
        Tensor operator+(double other) const;
        Tensor operator*(const Tensor& other) const;
        Tensor operator*(double other) const;
        Tensor operator-(const Tensor& other) const;
        Tensor operator-(double other) const;
        Tensor operator/(const Tensor& other) const;
        Tensor operator/(double other) const;
        Tensor operator-() const;
        Tensor operator[](int64_t index) const;
        Tensor operator[](const TensorIndices& indices) const;

        Tensor& operator+=(const Tensor& other);
        Tensor& operator+=(double other);
        Tensor& operator-=(const Tensor& other);
        Tensor& operator-=(double other);
        Tensor& operator*=(const Tensor& other);
        Tensor& operator*=(double other);
        Tensor& operator/=(const Tensor& other);
        Tensor& operator/=(double other);
        
        Tensor operator==(const Tensor& other) const;
        Tensor operator!=(const Tensor& other) const;
        Tensor operator<(const Tensor& other) const;
        Tensor operator<=(const Tensor& other) const;
        Tensor operator>(const Tensor& other) const;
        Tensor operator>=(const Tensor& other) const;
        Tensor operator==(double other) const;
        Tensor operator!=(double other) const;
        Tensor operator<(double other) const;
        Tensor operator<=(double other) const;
        Tensor operator>(double other) const;
        Tensor operator>=(double other) const;

        Tensor add(const Tensor& other) const;
        Tensor add(double other) const;
        Tensor sub(const Tensor& other) const;
        Tensor sub(double other) const;
        Tensor mul(const Tensor& other) const;
        Tensor mul(double other) const;
        Tensor pow(const Tensor& exponent) const;
        Tensor pow(double exponent) const;
        Tensor div(const Tensor& other) const;
        Tensor div(double other) const;
        Tensor matmul(const Tensor& other) const;
        Tensor neg() const;

        Tensor& requires_grad_(bool _requires_grad);
        Tensor& add_(const Tensor& other);
        Tensor& add_(double other);
        Tensor& sub_(const Tensor& other);
        Tensor& sub_(double other);
        Tensor& mul_(const Tensor& other);
        Tensor& mul_(double other);
        Tensor& pow_(const Tensor& exponent);
        Tensor& pow_(double exponent);
        Tensor& div_(const Tensor& other);
        Tensor& div_(double other);
        Tensor& neg_();
        Tensor& zero_();
        Tensor& fill_(double value);
        Tensor& uniform_(double from = 0, double to = 1);
        Tensor& normal_(double mean = 0, double std = 1);
        Tensor& unsqueeze_(int64_t dim);
        Tensor& scatter_(int64_t dim, const Tensor& index, double value);
        Tensor& index_put_(const TensorIndices& idx, const Tensor& value);
        Tensor& index_put_(const TensorIndices& idx, double value);
        Tensor& copy_(const Tensor& src);
        Tensor& to_(torch::Device device);
        Tensor& to_(torch::Dtype dtype);

        Tensor rsub(const Tensor& other) const;
        Tensor rsub(double other) const;
        Tensor rpow(const Tensor& exponent) const;
        Tensor rpow(double exponent) const;
        Tensor rdiv(const Tensor& exponent) const;
        Tensor rdiv(double exponent) const;
        Tensor rmatmul(const Tensor& exponent) const;
        
        Tensor to(torch::Device device) const;
        Tensor to(torch::Dtype dtype) const;
        Tensor cpu() const;
        Tensor cuda(int8_t index = 0) const;

        Tensor clone() const;
        Tensor exp() const;
        Tensor log() const;
        Tensor sqrt() const;
        Tensor sum(torch::IntArrayRef dim = {}, bool keepdim = false) const;
        Tensor mean(torch::IntArrayRef dim = {}, bool keepdim = false) const;
        Tensor max() const;
        std::tuple<Tensor, Tensor> max(int64_t dim, bool keepdim = false) const;
        Tensor min() const;
        std::tuple<Tensor, Tensor> min(int64_t dim, bool keepdim = false) const;
        Tensor squeeze(torch::IntArrayRef dim = {}) const;
        Tensor unsqueeze(int64_t dim) const;
        Tensor reshape(torch::IntArrayRef shape) const;
        Tensor broadcast_to(torch::IntArrayRef shape) const;
        Tensor transpose(int64_t dim0, int64_t dim1) const;
        Tensor t() const;
        Tensor swapdims(int64_t dim0, int64_t dim1) const;
        Tensor movedim(int64_t src, int64_t dest) const;
        Tensor slice(const TensorIndices& indices) const;
        Tensor select(int64_t dim, int64_t index) const;
        Tensor where(const Tensor& condition, const Tensor& other) const;
        Tensor where(const Tensor& condition, double other) const;

        Tensor relu() const;
        Tensor sigmoid() const;
        Tensor softmax(int64_t dim) const;
        Tensor log_softmax(int64_t dim) const;
        Tensor flatten(int64_t start_dim = 0, int64_t end_dim = -1) const;


        void set_output_nr(uint32_t nr);
        uint32_t output_nr() const;

        TensorIterator begin() const;
        TensorIterator end() const;
    
        std::string to_string() const;
        static std::string to_string(torch::Tensor tensor);

    private:
        struct Impl;
        std::shared_ptr<Impl> impl_;
    };

    Tensor operator+(double scalar, const Tensor& tensor);
    Tensor operator*(double scalar, const Tensor& tensor);
    Tensor operator-(double scalar, const Tensor& tensor);
    Tensor operator/(double scalar, const Tensor& tensor);

    class SYNAPX_API TensorIterator {
    private:
        Tensor tensor;
        int64_t index;

    public:
        TensorIterator(Tensor t, int64_t i) : tensor(t), index(i) {}
        Tensor operator*() const { return tensor[index]; }
        TensorIterator& operator++() { ++index; return *this; }
        bool operator==(const TensorIterator& other) const { return index == other.index;}
        bool operator!=(const TensorIterator& other) const { return index != other.index; }
    };

} // namespace synapx

#endif // TENSOR_HPP
