#ifndef FUNCTIONAL_HPP
#define FUNCTIONAL_HPP

#include <synapx/core.hpp>
#include <synapx/tensor.hpp>

namespace synapx
{
namespace F
{

SYNAPX_API Tensor add(const Tensor& t1, const Tensor& t2);

// tensor mul(const tensor& t1, const tensor& t2);

// tensor matmul(const tensor& t1, const tensor& t2);

} // namespace F

} // namespace synapzx

#endif // FUNCTIONAL_HPP