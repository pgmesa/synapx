from __future__ import annotations
import numpy
__all__ = ['Tensor', 'from_numpy', 'ones', 'zeros']
class Tensor:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: list[int]) -> None:
        ...
    def add(self, arg0: Tensor) -> Tensor:
        ...
    def matmul(self, arg0: Tensor) -> Tensor:
        ...
    def mul(self, arg0: Tensor) -> Tensor:
        ...
    def numpy(self) -> numpy.ndarray[numpy.float32]:
        ...
def from_numpy(arg0: numpy.ndarray[numpy.float32]) -> ...:
    """
    Create a tensor from a numpy array
    """
def ones(arg0: list[int]) -> ...:
    """
    Create a tensor filled with ones
    """
def zeros(arg0: list[int]) -> ...:
    """
    Create a tensor filled with zeros
    """
