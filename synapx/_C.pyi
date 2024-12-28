"""
Synapx tensor operations
"""
from __future__ import annotations
import numpy
__all__ = ['Tensor', 'from_numpy', 'matmul', 'ones', 'zeros']
class Tensor:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: ...) -> None:
        ...
    def matmul(self, arg0: Tensor) -> Tensor:
        ...
    def ndim(self) -> int:
        ...
    def numel(self) -> int:
        ...
    def numpy(self) -> numpy.ndarray:
        """
        Convert Tensor to NumPy array
        """
def from_numpy(arg0: numpy.ndarray) -> Tensor:
    """
    Create a tensor from numpy array
    """
def matmul(arg0: Tensor, arg1: Tensor) -> Tensor:
    """
    Matmul between two tensors
    """
def ones(arg0: list) -> Tensor:
    """
    Create a tensor filled with ones
    """
def zeros(arg0: list) -> Tensor:
    """
    Create a tensor filled with zeros
    """
