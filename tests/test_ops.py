
import torch
import synapx
import numpy as np

from utils import check_tensors, time_fun


atol = 1e-8; rtol = 1e-5

def op_tester(inputs:list, function, name, device='cpu', module_func=False, nn_functional=False, factor=1, offset=0, backward=True):    
    torch_inputs = [
        torch.tensor(np.random.rand(*shape)*factor+offset, requires_grad=True, dtype=torch.float32, device=device) 
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
        synapx.tensor(inp.detach().numpy(), requires_grad=True, dtype=torch.float32, device=device) 
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
    
    assert check_tensors(syn_out.torch(), torch_out, atol=atol, rtol=rtol)
    if backward:
        for synap_inp, torch_inp in zip(syn_inputs, torch_inputs):
            assert check_tensors(synap_inp.torch(), torch_inp, atol=atol, rtol=rtol)
            assert check_tensors(synap_inp.grad.torch(), torch_inp.grad, atol=atol, rtol=rtol)

# *************************
# ******* Basic ops *******
# *************************

def test_add():
    op_tester([(1000, 1500), (1000, 1500)], lambda x,y: x+y, name='add')

def test_sub():
    op_tester([(1000, 1500), (1000, 1500)], lambda x,y: x-y, name='sub')

def test_mul():
    op_tester([(2000, 2000), (2000, 2000)], lambda x,y: x*y, name='mul')

def test_div():
    op_tester([(500, 800), (500, 800)], lambda x,y: x/y, name='div')

def test_matmul():
    op_tester([(3, 512, 512), (3, 512, 512)], lambda x,y: x@y, name='matmul')
    
# def test_addmm():
#     op_tester([(10, 10), (10, 10), (10, 10)], lambda engine, x, y, z: engine.addmm(x, y, z), name='addmm', module_func=True)

def test_rsub():
    op_tester([(500, 400, 3)], lambda x: 234 - x, name='rsub')

def test_rmul():
    op_tester([(500, 400, 3)], lambda x: 6.4 * x, name='rmul')

def test_radd():
    op_tester([(500, 400, 3)], lambda x: 34 + x, name='radd')

def test_add_broadcast():
    op_tester([(1000, 1000), (1000, 1)], lambda x,y: x+y, name='add_broadcast')

def test_sub_broadcast():
    op_tester([(1000, 1000), (1000, 1)], lambda x,y: x-y, name='sub_broadcast')

def test_mul_broadcast():
    op_tester([(1000, 1000), (1000, 1)], lambda x,y: x*y, name='mul_broadcast')

def test_div_broadcast():
    op_tester([(1000, 1000), (1000, 1)], lambda x,y: x/y, name='div_broadcast')