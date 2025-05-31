"""
Synapx core C++ bindings
"""
from __future__ import annotations
import numpy
__all__ = ['Tensor', 'add', 'from_numpy', 'matmul', 'mul', 'ones', 'zeros']
class Tensor:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __add__(self, arg0: Tensor) -> Tensor:
        ...
    def __init__(self, arg0: ...) -> None:
        ...
    def __matmul__(self, arg0: Tensor) -> Tensor:
        ...
    def __mul__(self, arg0: Tensor) -> Tensor:
        ...
    def add(self, arg0: Tensor) -> Tensor:
        ...
    def dim(self) -> int:
        ...
    def matmul(self, arg0: Tensor) -> Tensor:
        ...
    def mul(self, arg0: Tensor) -> Tensor:
        ...
    def numel(self) -> int:
        ...
    def numpy(self) -> numpy.ndarray:
        """
        Convert Tensor to NumPy array
        """
    @property
    def shape(self) -> list[int]:
        ...
def add(arg0: Tensor, arg1: Tensor) -> Tensor:
    """
    Element-wise addition between two tensors
    """
def from_numpy(arg0: numpy.ndarray) -> Tensor:
    """
    Create a tensor from numpy array
    """
def matmul(arg0: Tensor, arg1: Tensor) -> Tensor:
    """
    Matmul between two tensors
    """
def mul(arg0: Tensor, arg1: Tensor) -> Tensor:
    """
    Element-wise product between two tensors
    """
def ones(arg0: list) -> Tensor:
    """
    Create a tensor filled with ones
    """
def zeros(arg0: list) -> Tensor:
    """
    Create a tensor filled with zeros
    """
