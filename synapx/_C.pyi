"""
Synapx tensor operations
"""
from __future__ import annotations
__all__ = ['Tensor', 'from_torch', 'ones', 'zeros']
class Tensor:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: ...) -> None:
        ...
    def ndim(self) -> int:
        ...
    def numel(self) -> int:
        ...
    def torch(self) -> ...:
        ...
def from_torch(arg0: ...) -> Tensor:
    """
    Create a tensor from PyTorch tensor
    """
def ones(arg0: list) -> Tensor:
    """
    Create a tensor filled with ones
    """
def zeros(arg0: list) -> Tensor:
    """
    Create a tensor filled with zeros
    """
