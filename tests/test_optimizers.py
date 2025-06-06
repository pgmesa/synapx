
import torch
import synapx

import synapx.optim
from utils import check_tensors


def check_optimizer(opt: synapx.optim.Optimizer, param: synapx.Tensor, 
                    opt_t:torch.optim.Optimizer, param_t: torch.Tensor):
    inp = synapx.ones((4,4))
    inp_t = torch.ones((4,4))
    
    for _ in range(3):
        # synapx
        out = inp @ param
        opt.zero_grad()
        out.sum().backward()
        opt.step()

        # torch
        out_t = inp_t @ param_t
        opt_t.zero_grad()
        out_t.sum().backward()
        opt_t.step()
        
    print("Param:", param)
    print("Param:", param_t)
    
    assert check_tensors(inp, inp_t)
    assert check_tensors(param, param_t)
    assert check_tensors(inp.grad, inp_t.grad)
    assert check_tensors(out, out_t)
    assert check_tensors(param, param_t)
    assert check_tensors(param.grad, param_t.grad)


def test_SGD():
    attrs = {
        "lr": 0.1,
        "momentum": 0.9,
        "maximize": False,
        "dampening": 0,
        "nesterov": True,
        "weight_decay": 0.5,
    }
    
    param = synapx.ones((4,4), requires_grad=True)
    opt = synapx.optim.SGD([param], **attrs)
    
    param_t = torch.ones((4,4), requires_grad=True)
    opt_t = torch.optim.SGD([param_t], **attrs)
    
    check_optimizer(opt, param, opt_t, param_t)
    
    
def test_Adam():
    attrs = {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "weight_decay": 0.7,
        "maximize": False
    }
    
    param = synapx.ones((4,4), requires_grad=True)
    opt = synapx.optim.Adam([param], **attrs)
    
    param_t = torch.ones((4,4), requires_grad=True)
    opt_t = torch.optim.Adam([param_t], **attrs)
    
    check_optimizer(opt, param, opt_t, param_t)
    
    
def test_AdamW():
    attrs = {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "weight_decay": 0.7,
        "maximize": False
    }
    
    param = synapx.ones((4,4), requires_grad=True)
    opt = synapx.optim.AdamW([param], **attrs)
    
    param_t = torch.ones((4,4), requires_grad=True)
    opt_t = torch.optim.AdamW([param_t], **attrs)
    
    check_optimizer(opt, param, opt_t, param_t)