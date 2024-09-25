
from typing import List, Union

class DataType:
    """
    Enum representing the supported data types for a Tensor.
    
    Values:
        UINT8: Unsigned 8-bit integer data type.
        INT32: Signed 32-bit integer data type.
        FLOAT32: 32-bit floating-point data type.
    """

    UINT8: 'DataType'
    """Unsigned 8-bit integer."""

    INT32: 'DataType'
    """Signed 32-bit integer."""

    FLOAT32: 'DataType'
    """32-bit floating-point."""

class Tensor:
    """
    Tensor class representing a multi-dimensional array with support for various data types.
    
    Attributes:
        numel: int
            The total number of elements in the tensor.
        
        shape: List[int]
            The shape of the tensor as a list of integers, where each integer represents the size of a dimension.
        
        ndim: int
            The number of dimensions of the tensor (also known as the rank).
        
        dtype: DataType
            The data type of the elements in the tensor, such as DataType.UINT8, DataType.INT32, or DataType.FLOAT32.
        
        strides: List[int]
            The memory strides for each dimension of the tensor, determining how far apart consecutive elements in memory are.
    """

    numel: int
    """The total number of elements in the tensor."""

    shape: List[int]
    """The shape of the tensor as a list of integers, where each integer represents the size of a dimension."""

    ndim: int
    """The number of dimensions (rank) of the tensor."""

    dtype: DataType
    """The data type of the elements in the tensor."""

    strides: List[int]
    """The memory strides for each dimension of the tensor."""

    def __init__(self) -> None:
        """
        Initialize an empty Tensor.
        
        A default constructor that initializes a tensor with no specific shape or data.
        """
        ...

    def __repr__(self) -> str:
        """
        Return the string representation of the Tensor.
        
        This function provides a human-readable summary of the tensor, including its shape, data type, and number of elements.
        
        Returns:
            str: The string representation of the tensor.
        """
        ...

    def __getitem__(self, index: Union[int, slice]) -> 'Tensor':
        """
        Get an element or a slice from the Tensor.
        
        Allows indexing or slicing of the tensor to access specific elements or sub-tensors.
        
        Args:
            index: int or slice
                The index or slice to retrieve.
        
        Returns:
            Tensor: A new tensor representing the element or slice.
        """
        ...

    def __add__(self, other: Union['Tensor', float]) -> 'Tensor':
        """
        Add another Tensor or a scalar to this Tensor.
        
        Supports element-wise addition between tensors or broadcasting addition of a scalar to all elements.
        
        Args:
            other: Tensor or float
                The tensor or scalar to add.
        
        Returns:
            Tensor: A new tensor representing the result of the addition.
        """
        ...

    def __iadd__(self, other: Union['Tensor', float]) -> 'Tensor':
        """
        In-place addition of another Tensor or a scalar to this Tensor.
        
        Modifies the current tensor by adding the values from another tensor or a scalar.
        
        Args:
            other: Tensor or float
                The tensor or scalar to add.
        
        Returns:
            Tensor: The modified tensor after addition.
        """
        ...

    def __mul__(self, other: Union['Tensor', float]) -> 'Tensor':
        """
        Multiply this Tensor by another Tensor or a scalar.
        
        Supports element-wise multiplication between tensors or broadcasting multiplication of a scalar.
        
        Args:
            other: Tensor or float
                The tensor or scalar to multiply by.
        
        Returns:
            Tensor: A new tensor representing the result of the multiplication.
        """
        ...

    def __imul__(self, other: Union['Tensor', float]) -> 'Tensor':
        """
        In-place multiplication of this Tensor by another Tensor or a scalar.
        
        Modifies the current tensor by multiplying the values by another tensor or scalar.
        
        Args:
            other: Tensor or float
                The tensor or scalar to multiply by.
        
        Returns:
            Tensor: The modified tensor after multiplication.
        """
        ...

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """
        Matrix multiplication of this Tensor with another Tensor.
        
        Performs matrix multiplication (dot product) between two tensors. This corresponds to the `@` operator in Python.
        
        Args:
            other: Tensor
                The tensor to perform matrix multiplication with.
        
        Returns:
            Tensor: A new tensor representing the result of the matrix multiplication.
        """
        ...

    @staticmethod
    def matmul(a: 'Tensor', b: 'Tensor') -> 'Tensor':
        """
        Static method for matrix multiplication between two Tensors.
        
        Performs matrix multiplication (dot product) between two tensors.
        
        Args:
            a: Tensor
                The left-hand side tensor.
            
            b: Tensor
                The right-hand side tensor.
        
        Returns:
            Tensor: A new tensor representing the result of the matrix multiplication.
        """
        ...

    @staticmethod
    def empty(shape: List[int]) -> 'Tensor':
        """
        Create an uninitialized Tensor with a given shape.
        
        This method does not initialize the tensor with any specific values, which is useful for performance in some cases.
        
        Args:
            shape: List[int]
                The shape of the new tensor.
        
        Returns:
            Tensor: A new uninitialized tensor with the specified shape.
        """
        ...

    @staticmethod
    def fill(shape: List[int], dtype: DataType, value: float) -> 'Tensor':
        """
        Create a Tensor filled with a constant value.
        
        Args:
            shape: List[int]
                The shape of the new tensor.
            
            dtype: DataType
                The data type of the tensor elements.
            
            value: float
                The constant value to fill the tensor with.
        
        Returns:
            Tensor: A new tensor filled with the specified value.
        """
        ...

    @staticmethod
    def ones(shape: List[int], dtype: DataType) -> 'Tensor':
        """
        Create a Tensor filled with ones.
        
        Args:
            shape: List[int]
                The shape of the new tensor.
            
            dtype: DataType
                The data type of the tensor elements.
        
        Returns:
            Tensor: A new tensor filled with ones.
        """
        ...

    @staticmethod
    def zeros(shape: List[int], dtype: DataType) -> 'Tensor':
        """
        Create a Tensor filled with zeros.
        
        Args:
            shape: List[int]
                The shape of the new tensor.
            
            dtype: DataType
                The data type of the tensor elements.
        
        Returns:
            Tensor: A new tensor filled with zeros.
        """
        ...
