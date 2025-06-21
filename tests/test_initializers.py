import torch
import synapx

from utils import check_tensors


def verify_tensor_properties(tensor, expected_shape, expected_dtype=torch.float32):
    assert isinstance(tensor, synapx.Tensor), f"Expected Tensor type, got {type(tensor)}"
    assert tensor.dtype == expected_dtype, f"Expected tensor dtype {expected_dtype}, got {tensor.dtype}"
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.data.shape}"

def test_tensor():
    # Test with default float dtype
    data = [[1.0, 2], [3, 4]]
    s_t = synapx.tensor(data)
    t_t = torch.tensor(data)  
    verify_tensor_properties(s_t, (2, 2))
    assert check_tensors(s_t, t_t)
    
    # Test with explicit dtype
    s_t = synapx.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    t_t = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    verify_tensor_properties(s_t, (2, 2), torch.int64)
    assert check_tensors(s_t, t_t)

def test_empty():
    # We can't test exact values for empty as they're undefined,
    # but we can verify the allocated memory has the right shape and type
    t = synapx.empty((2, 3))
    verify_tensor_properties(t, (2, 3))
    assert t.numel() == 6  # verify allocated size
    
    t = synapx.empty((2, 3), dtype=torch.float64)
    verify_tensor_properties(t, (2, 3), torch.float64)
    assert t.numel() == 6

def test_ones():
    t = synapx.ones((2, 3))
    verify_tensor_properties(t, (2, 3))
    assert torch.all(t.data == 1)  # verify all elements are 1
    
    t = synapx.ones((2, 3), dtype=torch.float64)
    verify_tensor_properties(t, (2, 3), torch.float64)
    assert torch.all(t.data == 1)
    assert t.dtype == torch.float64

def test_ones_like():
    base = synapx.ones((2, 3))
    t = synapx.ones_like(base)
    verify_tensor_properties(t, (2, 3))
    assert torch.all(t.data == 1)
    
    t = synapx.ones_like(base, dtype=torch.float64)
    verify_tensor_properties(t, (2, 3), torch.float64)
    assert torch.all(t.data == 1)

def test_zeros():
    t = synapx.zeros((2, 3))
    verify_tensor_properties(t, (2, 3))
    assert torch.all(t.data == 0)  # verify all elements are 0
    
    t = synapx.zeros((2, 3), dtype=torch.float64)
    verify_tensor_properties(t, (2, 3), torch.float64)
    assert torch.all(t.data == 0)

def test_zeros_like():
    base = synapx.zeros((2, 3))
    t = synapx.zeros_like(base)
    verify_tensor_properties(t, (2, 3))
    assert torch.all(t.data == 0)
    
    t = synapx.zeros_like(base, dtype=torch.float64)
    verify_tensor_properties(t, (2, 3), torch.float64)
    assert torch.all(t.data == 0)

# def test_arange():
#     t = synapx.arange(5)
#     verify_tensor_properties(t, (5,))
#     assert torch.equal(t.data, torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32))
    
#     t = synapx.arange(1, 5)
#     verify_tensor_properties(t, (4,))
#     assert torch.equal(t.data, torch.tensor([1, 2, 3, 4], dtype=torch.float32))
    
#     t = synapx.arange(0, 10, 2)
#     verify_tensor_properties(t, (5,))
#     assert torch.equal(t.data, torch.tensor([0, 2, 4, 6, 8], dtype=torch.float32))

def test_rand():
    t = synapx.rand((2, 3))
    verify_tensor_properties(t, (2, 3))
    assert torch.all((t.data >= 0) & (t.data <= 1))  # verify range [0, 1]
    
    t = synapx.rand((2, 3), dtype=torch.float64)
    verify_tensor_properties(t, (2, 3), torch.float64)
    assert torch.all((t.data >= 0) & (t.data <= 1))

def test_randn():
    t = synapx.randn((2, 3))
    verify_tensor_properties(t, (2, 3))
    
    t = synapx.randn((2, 3), dtype=torch.float64)
    verify_tensor_properties(t, (2, 3), torch.float64)

# def test_normal():
#     mean, std = 5, 2
#     t = synapx.normal(mean, std, (2, 3))
#     verify_tensor_properties(t, (2, 3))
    
#     t = synapx.normal(mean, std, (2, 3), dtype=torch.float64)
#     verify_tensor_properties(t, (2, 3), torch.float64)

# def test_randint():
#     low, high = 0, 10
#     t = synapx.randint(low, high, (2, 3))
#     verify_tensor_properties(t, (2, 3), torch.int32)
#     assert torch.all((t.data >= low) & (t.data < high))  # verify range
#     assert torch.all(t.data.int() == t.data)  # verify integers
    
#     t = synapx.randint(low, high, (2, 3), dtype=torch.int64)
#     verify_tensor_properties(t, (2, 3), torch.int64)
#     assert torch.all((t.data >= low) & (t.data < high))

# def test_eye():
#     t = synapx.eye(3)
#     verify_tensor_properties(t, (3, 3))
#     # Verify diagonal is 1 and rest is 0
#     assert torch.equal(t.data, torch.eye(3, dtype=torch.float32))
    
#     t = synapx.eye(3, dtype=torch.float64)
#     verify_tensor_properties(t, (3, 3), torch.float64)
#     assert torch.equal(t.data, torch.eye(3, dtype=torch.float64))