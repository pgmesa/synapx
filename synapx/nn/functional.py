
from synapx._C import _nn as _functional
from synapx._C._nn import *

import synapx
from synapx import Tensor


# Functions without a C++ implementation, composed from differentiable ops.
# min(0, x) is written as -relu(-x) and |x| as relu(x) + relu(-x).

SELU_ALPHA = 1.6732632423543772848170429916717
SELU_SCALE = 1.0507009873554804934193349852946


def _reduce(loss: Tensor, reduction: str) -> Tensor:
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    if reduction == 'none':
        return loss
    raise ValueError(f"'{reduction}' is not a valid value for reduction")


def tanh(x: Tensor) -> Tensor:
    # tanh(x) = 2 * sigmoid(2x) - 1
    return synapx.sigmoid(x * 2.0) * 2.0 - 1.0


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    # max(0, x) + slope * min(0, x)
    return synapx.relu(x) - synapx.relu(-x) * negative_slope


def selu(x: Tensor) -> Tensor:
    # scale * (max(0, x) + alpha * (exp(min(0, x)) - 1))
    negative = (synapx.exp(-synapx.relu(-x)) - 1.0) * SELU_ALPHA
    return (synapx.relu(x) + negative) * SELU_SCALE


def cross_entropy(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    class_dim = 0 if input.dim() == 1 else 1
    return nll_loss(log_softmax(input, class_dim), target, reduction)


def binary_cross_entropy_with_logits(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    # Stable form of bce(sigmoid(x), t): max(x, 0) - x * t + log(1 + exp(-|x|)).
    # exp never overflows because its argument is always <= 0.
    abs_x = synapx.relu(input) + synapx.relu(-input)
    loss = synapx.relu(input) - input * target + synapx.log(synapx.exp(-abs_x) + 1.0)
    return _reduce(loss, reduction)
