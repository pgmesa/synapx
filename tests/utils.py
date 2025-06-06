
import warnings
from time import time
from typing import Union

import torch
import synapx


def time_fun(function, *args, **kwargs):
    t0 = time()
    out = function(*args, **kwargs)
    tf = time()
    
    return out, tf - t0 

Tensor = Union[torch.Tensor, synapx.Tensor]

def check_tensors(t1: Tensor, t2: Tensor, atol=1e-5, rtol=1e-4) -> bool:
    """Returns if 2 tensors have the same values and shape

    Args:
        t1 (Tensor): Tensor1
        t2 (Tensor): Tensor2
    """
    if isinstance(t1, synapx.Tensor):
        t1 = t1.torch()
        
    if isinstance(t2, synapx.Tensor):
        t2 = t2.torch()
    
    if t1 is None or t2 is None: 
        return t1 is t2
    
    if t1.dtype != t2.dtype:
        if not (t1.dtype.is_floating_point and t2.dtype.is_floating_point):
            warnings.warn(f"\ndifferent floating types t1={t1.dtype} t2={t2.dtype}", stacklevel=0) 
        else:
            warnings.warn(f"\ndtype of tensors don't match t1={t1.dtype} t2={t2.dtype}", stacklevel=0) 
        t1 = t1.type(t2.dtype)
    
    check = torch.equal(t1, t2) or torch.allclose(t1, t2, rtol=rtol, atol=atol)
    
    if not check:
        print(f"Shapes {t1.shape} {t2.shape}")
        print("Max Abs error:", (t1 - t2).abs().max().item())
        print("Max Rel error:", (t1 - t2).abs().max().item() / (t2.abs().max().item()))
    
    return check