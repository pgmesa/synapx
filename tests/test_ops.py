
import pytest

import torch
import synapx
import numpy as np

from utils import check_tensors, time_fun


atol = 1e-8; rtol = 1e-5

def op_tester(inputs:list, function, name, device='cpu', module_func=False,
              nn_functional=False, factor=1, offset=0, backward=True, ones=False, dtype=torch.float32):    
    if ones:
        torch_inputs = [
            torch.tensor(np.ones(shape)*factor+offset, requires_grad=backward, dtype=dtype, device=device) 
            for shape in inputs
        ]
    else:
        torch_inputs = [
            torch.tensor(np.random.rand(*shape)*factor+offset, requires_grad=backward, dtype=dtype, device=device) 
            for shape in inputs
        ]
        
    if module_func: 
        if nn_functional:
            torch_inputs.insert(0, torch.nn.functional)
        else:
            torch_inputs.insert(0, torch)
    torch_out, torch_fw_time = time_fun(function, *torch_inputs)
    if not isinstance(torch_out, torch.Tensor):
        torch_out = torch_out[0]
    if backward:
        _, torch_bw_time = time_fun(torch_out.backward, torch.ones_like(torch_out))
    
    torch_inputs = torch_inputs[1:] if module_func else torch_inputs
    
    syn_inputs = [
        synapx.tensor(inp.detach(), requires_grad=inp.requires_grad, dtype=inp.dtype, device=device) 
        for inp in torch_inputs
    ]
    if module_func: 
        if nn_functional:
            syn_inputs.insert(0, synapx.nn.functional)
        else:
            syn_inputs.insert(0, synapx)
    syn_out, syn_fw_time = time_fun(function, *syn_inputs)
    if not isinstance(syn_out, synapx.Tensor):
        syn_out = syn_out[0]
    if backward:
        _, syn_bw_time = time_fun(syn_out.backward, synapx.ones_like(syn_out))
    
    syn_inputs = syn_inputs[1:] if module_func else syn_inputs 
    
    if backward:
        print(f'\n{name},  device: {device},  torch/synapx ' + 
                f'fp: {torch_fw_time*1000:.2f} / {syn_fw_time*1000:.2f} ms, ' + 
                f'bp: {torch_bw_time*1000:.2f} / {syn_bw_time*1000:.2f} ms')
    else:
        print(f'\n{name},  device: {device},  torch/synapx ' + 
                f'fp: {torch_fw_time*1000:.2f} / {syn_fw_time*1000:.2f} ms')
    
    assert check_tensors(syn_out, torch_out, atol=atol, rtol=rtol)
    if backward:
        for synap_inp, torch_inp in zip(syn_inputs, torch_inputs):
            assert check_tensors(synap_inp, torch_inp, atol=atol, rtol=rtol)
            assert check_tensors(synap_inp.grad, torch_inp.grad, atol=atol, rtol=rtol)

# *************************
# ******* Basic ops *******
# *************************

def test_add():
    op_tester([(1000, 1500), (1000, 1500)], lambda x, y: x + y, name='add+')
    op_tester([(1000, 1000), (1000, 1)], lambda x, y: x + y, name='add_broadcast')
    op_tester([(1000, 1500), (1000, 1500)], lambda x, y: x.add(y), name='tensor.add()')
    op_tester([(1000, 1500), (1000, 1500)], lambda engine, x, y: engine.add(x, y), name='engine.add()', module_func=True)
    op_tester([(1000, 1500)], lambda x: x.add(-50), name='tensor.add(scalar)')
    op_tester([(1000, 1500)], lambda engine, x: engine.add(x, 50), name='engine.add(scalar)', module_func=True)

def test_sub():
    op_tester([(1000, 1500), (1000, 1500)], lambda x, y: x - y, name='sub')
    op_tester([(1000, 1000), (1000, 1)], lambda x, y: x - y, name='sub_broadcast')
    op_tester([(1000, 1500), (1000, 1500)], lambda x, y: x.sub(y), name='tensor.sub()')
    op_tester([(1000, 1500), (1000, 1500)], lambda engine, x, y: engine.sub(x, y), name='engine.sub()', module_func=True)
    op_tester([(1000, 1500)], lambda x: x.sub(50), name='tensor.sub(scalar)')
    op_tester([(1000, 1500)], lambda engine, x: engine.sub(x, 50), name='engine.sub(scalar)', module_func=True)

def test_mul():
    op_tester([(2000, 2000), (2000, 2000)], lambda x, y: x * y, name='mul')
    op_tester([(1000, 1000), (1000, 1)], lambda x, y: x * y, name='mul_broadcast')
    op_tester([(1000, 1500), (1000, 1500)], lambda x, y: x.mul(y), name='tensor.mul()')
    op_tester([(1000, 1500), (1000, 1500)], lambda engine, x, y: engine.mul(x, y), name='engine.mul()', module_func=True)
    op_tester([(1000, 1500)], lambda x: x.mul(2), name='tensor.mul(scalar)')
    op_tester([(1000, 1500)], lambda engine, x: engine.mul(x, 2), name='engine.mul(scalar)', module_func=True)

def test_div():
    op_tester([(500, 800), (500, 800)], lambda x, y: x / y, name='div')
    op_tester([(1000, 1000), (1000, 1)], lambda x, y: x / y, name='div_broadcast')
    op_tester([(1000, 1500), (1000, 1500)], lambda x, y: x.div(y), name='tensor.div()')
    op_tester([(1000, 1500), (1000, 1500)], lambda engine, x, y: engine.div(x, y), name='engine.div()', module_func=True)
    op_tester([(1000, 1500)], lambda x: x.div(2), name='tensor.div(scalar)')
    op_tester([(1000, 1500)], lambda engine, x: engine.div(x, 2), name='engine.div(scalar)', module_func=True)

def test_matmul():
    op_tester([(3, 512, 512), (3, 512, 512)], lambda x, y: x @ y, name='matmul')
    op_tester([(1000, 1500), (1500, 2000)], lambda x, y: x.matmul(y), name='tensor.matmul()')
    op_tester([(1000, 1500), (1500, 2000)], lambda engine, x, y: engine.matmul(x, y), name='engine.matmul()', module_func=True)

def test_pow():
    op_tester([(1000, 1000)], lambda x: x ** 1.1, name='pow')
    op_tester([(1000, 1000)], lambda x: x ** -2.2, name='pow')
    op_tester([(1000, 1000)], lambda x: x ** 0.7, name='pow')
    op_tester([(1000, 1500)], lambda x: x.pow(2), name='tensor.pow(scalar)')
    op_tester([(1000, 1500)], lambda engine, x: engine.pow(x, 2), name='engine.pow(scalar)', module_func=True)
    
def test_addmm():
    op_tester([(10, 10), (10, 10), (10, 10)], lambda engine, x, y, z: engine.addmm(x, y, z), name='addmm', module_func=True)
    

def test_radd():
    op_tester([(500, 400, 3)], lambda x: 34 + x, name='radd')

def test_rsub():
    op_tester([(500, 400, 3)], lambda x: 234 - x, name='rsub')

def test_rmul():
    op_tester([(500, 400, 3)], lambda x: 6.4 * x, name='rmul')
    
def test_rpow():
    op_tester([(1000, 1000)], lambda x: -2 ** x, name='rpow')

def test_neg():
    op_tester([(1000, 1500)], lambda x: -x, name='neg')
    
    
def test_iadd():
    a_synapx = synapx.ones((4, 4))
    a_torch = torch.ones((4, 4)) 
    
    def op(inp, b):
        inp += 2
        inp.add_(2)
        inp += b
        inp.add_(b)
        inp += inp
    
    prev_id = id(a_synapx)
    op(a_synapx, synapx.ones((4, 4)))
    op(a_torch, torch.ones((4, 4)))

    assert prev_id == id(a_synapx)
    assert check_tensors(a_synapx, a_torch)

def test_isub():
    a_synapx = synapx.ones((4, 4))
    a_torch = torch.ones((4, 4))
    b = synapx.ones((4, 4))

    def op(inp, b):
        inp -= 1
        inp.sub_(1)
        inp -= b
        inp.sub_(b)
        inp -= inp * 2
    
    prev_id = id(a_synapx)
    op(a_synapx, b)
    op(a_torch, b.torch())

    assert prev_id == id(a_synapx)
    assert check_tensors(a_synapx, a_torch)

def test_imul():
    a_synapx = synapx.ones((4, 4))
    a_torch = torch.ones((4, 4))
    b = synapx.ones((4, 4))

    def op(inp, b):
        inp *= 2
        inp.mul_(2)
        inp *= b
        inp.mul_(b)
        inp *= inp
    
    prev_id = id(a_synapx)
    op(a_synapx, b)
    op(a_torch, b.torch())

    assert prev_id == id(a_synapx)
    assert check_tensors(a_synapx, a_torch)

def test_idiv():
    a_synapx = synapx.ones((4, 4)) * 4
    a_torch = torch.ones((4, 4)) * 4
    b = synapx.ones((4, 4)) * 2

    def op(inp, b):
        inp /= 2
        inp.div_(2)
        inp /= b
        inp.div_(b)
        inp /= inp
    
    prev_id = id(a_synapx)
    op(a_synapx, b)
    op(a_torch, b.torch())

    assert prev_id == id(a_synapx)
    assert check_tensors(a_synapx, a_torch)

def test_ineg():
    a = synapx.ones((4, 4))

    prev_id = id(a)
    a.neg_()

    assert prev_id == id(a)
    assert (a.sum().item() == -1 * a.numel())

def test_zerofill():
    a = synapx.ones((4, 4))

    prev_id = id(a)
    a.zero_()

    assert prev_id == id(a)
    assert (a.sum().item() == 0)

def test_ipow():
    a = synapx.ones((4, 4))

    prev_id = id(a)
    a.pow_(2)

    assert prev_id == id(a)
    assert (a.sum().item() == 1 * a.numel())
    
def test_parameter_update():
    a = synapx.ones((4, 4), requires_grad=True)

    with synapx.no_grad():
        a += 2.0  # This should work

    with pytest.raises(RuntimeError):
        a /= 2  # This should raise a RuntimeError

    
# # *************************
# # ******* Other ops *******
# # *************************
    
# def test_slice():
#     op_tester([(30, 40, 20, 10)], lambda x: x[10:14, 3:, :, :], name='slice')
#     op_tester([(30, 40, 20, 10)], lambda x: x[10:25, :5, :12, 4:], name='slice')
#     op_tester([(30, 40, 20, 10)], lambda x: x[10:11, 4:8, :12, 0:-1], name='slice')

def test_clone():
    op_tester([(1000, 1000)], lambda x: x.clone(), name='clone')

def test_log():
    op_tester([(1000, 1000)], lambda x: x.log(), name='log', factor=5, offset=0.1)

def test_exp():
    op_tester([(1000, 1000)], lambda x: x.exp(), name='exp')

def test_sqrt():
    op_tester([(1000, 1000)], lambda x: x.sqrt(), name='sqrt')

def test_sum():
    op_tester([(1000, 1500)], lambda x: x.sum(), name='sum')
    op_tester([(1000, 1500)], lambda x: x.sum(dim=1), name='sum')
    op_tester([(1000, 1500, 3)], lambda x: x.sum(dim=(-2, 2)), name='sum')
    op_tester([(1000, 1500)], lambda x: x.sum(dim=-1), name='sum')
    op_tester([(1000, 1500, 3)], lambda engine, x: engine.sum(x, dim=-1), name='engine.sum', module_func=True)

def test_mean():
    op_tester([(1000, 1500)], lambda x: x.mean(dim=0), name='mean')
    op_tester([(1000, 1500)], lambda x: x.mean(dim=1), name='mean')
    op_tester([(1000, 1500)], lambda x: x.mean(dim=-1), name='mean')
    op_tester([(1000, 1500, 3)], lambda engine, x: engine.mean(x, dim=(1,2)), name='engine.mean', module_func=True)

def test_max():
    op_tester([(4, 4)], lambda x: x.max(), name='max')
    op_tester([(4, 4)], lambda x: x.max(), name='max_ones', ones=True) # When more than one element is equal to max_value
    op_tester([(1000, 1500)], lambda x: x.max(dim=0), name='max_d0')
    op_tester([(1000, 1500)], lambda x: x.max(dim=0), name='max_d0_ones', ones=True)
    op_tester([(1000, 1500)], lambda x: x.max(dim=1), name='max_d1')
    op_tester([(1000, 1500)], lambda x: x.max(dim=1), name='max_d1_ones', ones=True)
    op_tester([(1000, 1500)], lambda x: x.max(dim=1, keepdim=True), name='max_d1')
    op_tester([(1000, 3, 4, 5)], lambda x: x.max(dim=2), name='max_d2')
    op_tester([(1000, 3, 4, 5)], lambda x: x.max(dim=3), name='max_d3')
    op_tester([(1000, 3, 4, 5)], lambda x: x.max(dim=-1), name='max_d3')
    op_tester([(1000, 3, 4, 5)], lambda x: x.max(dim=-2), name='max_d3')
    
def test_min():
    op_tester([(4, 4)], lambda x: x.min(), name='min')
    op_tester([(4, 4)], lambda x: x.min(), name='min_ones', ones=True) # When more than one element is equal to min_value
    op_tester([(1000, 1500)], lambda x: x.min(dim=0), name='min_d0')
    op_tester([(1000, 1500)], lambda x: x.min(dim=0), name='min_d0_ones', ones=True)
    op_tester([(1000, 1500)], lambda x: x.min(dim=1), name='min_d1')
    op_tester([(1000, 1500)], lambda x: x.min(dim=1), name='min_d1_ones', ones=True)
    op_tester([(1000, 1500)], lambda x: x.min(dim=1, keepdim=True), name='min_d1')
    op_tester([(1000, 3, 4, 5)], lambda x: x.min(dim=2), name='min_d2')
    op_tester([(1000, 3, 4, 5)], lambda x: x.min(dim=3), name='min_d3')
    op_tester([(1000, 3, 4, 5)], lambda x: x.min(dim=-1), name='min_d3')
    op_tester([(1000, 3, 4, 5)], lambda x: x.min(dim=-2), name='min_d3')

def test_squeeze():
    op_tester([(100, 1)], lambda x: x.squeeze(dim=1), name='squeeze')
    op_tester([(1, 3)], lambda x: x.squeeze(dim=0), name='squeeze')
    op_tester([(1, 100, 1, 2)], lambda x: x.squeeze(), name='squeeze')
    
def test_unsqueeze():
    op_tester([(1000,)], lambda x: x.unsqueeze(dim=1), name='unsqueeze')
    op_tester([(1000, 23)], lambda x: x.unsqueeze(dim=1), name='unsqueeze')

# def test_reshape():
#     op_tester([(100000,)], lambda x: x.reshape((1000, 100)), name='reshape')
#     op_tester([(10000, 200)], lambda x: x.reshape((-1, 50)), name='reshape')
    
# def test_movedim():
#     op_tester([(100, 200, 300)], lambda x: x.movedim(0, 1), name='movedim')
#     op_tester([(10000, 200)], lambda x: x.movedim(-1, -2), name='movedim')
    
# def test_flatten():
#     op_tester([(100, 200, 300)], lambda x: x.flatten(), name='flatten')
#     op_tester([(100, 200, 300)], lambda x: x.flatten(start_dim=0, end_dim=1), name='flatten_0_1')
#     op_tester([(100, 200, 300)], lambda x: x.flatten(start_dim=1, end_dim=2), name='flatten_1_2')
#     op_tester([(100, 200, 300)], lambda x: x.flatten(start_dim=0, end_dim=2), name='flatten_0_2')

# def test_transpose():
#     op_tester([(100, 200, 300)], lambda x: x.transpose(2, 1), name='transpose')
#     op_tester([(100, 200, 300)], lambda x: x.transpose(0, 1), name='transpose')

# # **********************************
# # ******* Array manipulation *******
# # **********************************

# def test_stack():
#     op_tester([(100,), (100,), (100,)], lambda engine, *x: engine.stack(x, dim=0), name='stack', module_func=True)
#     op_tester([(100, 100), (100, 100), (100, 100)], lambda engine, *x: engine.stack(x, dim=1), name='stack', module_func=True)
#     op_tester([(100, 100, 100), (100, 100, 100), (100, 100, 100)], lambda engine, *x: engine.stack(x, dim=2), name='stack',module_func=True)

# def test_concat():
#     op_tester([(100, 200, 4), (100, 200, 3)], lambda engine, *x: engine.concat(x, dim=-1), name='concat', module_func=True)
#     op_tester([(100, 200, 4), (22, 200, 4)], lambda engine, *x: engine.concat(x, dim=0), name='concat', module_func=True)

# def test_unbind():
#     op_tester([(100, 200, 300)], lambda engine, x: engine.unbind(x, dim=-1), name='unbind', module_func=True)
#     op_tester([(100, 200, 300)], lambda engine, x: engine.unbind(x, dim=1), name='unbind', module_func=True)

# # **************************
# # ******* Linear ops *******
# # **************************

# def test_linear():
#     op_tester([(100, 200), (300, 200)], lambda F, x, w: F.linear(x, w), name='linear', module_func=True, nn_functional=True)
#     op_tester([(100, 200), (300, 200)], lambda F, x, w: F.linear(x, w), name='linear', module_func=True, nn_functional=True)
#     op_tester([(100, 200), (300, 200), (100, 300)], lambda F, x, w, b: F.linear(x, w, b), name='linear', module_func=True, nn_functional=True)
