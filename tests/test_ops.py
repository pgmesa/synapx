
from typing import Callable

import pytest
import torch
import synapx
import numpy as np

from utils import check_tensors, time_fun


atol = 1e-6; rtol = 1e-5

def op_tester(
    inputs: tuple | dict, 
    function: Callable,
    name: str,
    device: str = 'cpu',
    module_func: bool = False,
    nn_functional: bool = False,
    backward: bool = True,
    input_callback: Callable = None,
    loss_function: bool = False,
    classification_target: bool = False,
    target_shape: tuple = None,
    reduction: str = 'mean'
):
    """    
    Args:
        inputs: List of dicts with keys: 'shape', 'dtype', 'initializer', 'requires_grad'
                Or legacy tuple format for backward compatibility
        function: Function to test
        name: Test name
        device: Device to run on
        module_func: Whether function expects module as first arg
        nn_functional: Whether to use torch.nn.functional
        backward: Whether to test backward pass
        input_callback: Custom function to create inputs
        loss_function: Whether this is a loss function (creates target automatically)
        classification_target: Whether the targets should be class indices
        target_shape: Shape for loss function target (if different from prediction)
        reduction: Reduction for loss functions ('mean', 'sum', 'none')
    """
    
    # Handle legacy tuple format
    if inputs and isinstance(inputs[0], (tuple, list)) and not isinstance(inputs[0], dict):
        inputs = [
            {'shape': shape, 'dtype': torch.float32, 'initializer': 'random', 'requires_grad': backward} 
            for shape in inputs
        ]
    
    def create_tensor(spec: dict, device: str, framework: str = 'torch'):
        shape = spec['shape']
        dtype = spec.get('dtype', torch.float32)
        initializer = spec.get('initializer', 'random')
        requires_grad = spec.get('requires_grad', True)
        
        # Create numpy array based on initializer
        if initializer == 'random':
            data = np.random.rand(*shape)
        elif initializer == 'randn':
            data = np.random.randn(*shape)
        elif initializer == 'ones':
            data = np.ones(shape)
        elif initializer == 'zeros':
            data = np.zeros(shape)
        elif callable(initializer):
            data = initializer(shape)
        else:
            data = np.random.rand(*shape)
        
        if framework == 'torch':
            return torch.tensor(data, requires_grad=requires_grad, dtype=dtype, device=device)
        else:  # synapx
            return synapx.tensor(data, requires_grad=requires_grad, dtype=dtype, device=device)
    
    # Create torch inputs
    if input_callback:
        torch_inputs = input_callback(inputs, device, 'torch')
    else:
        torch_inputs = [create_tensor(spec, device, 'torch') for spec in inputs]
    
    # Handle loss functions - add target tensor
    if loss_function:
        pred_shape = torch_inputs[0].shape
        tgt_shape = target_shape or pred_shape
        
        if reduction == 'none':
            # For classification losses, create class indices
            if len(tgt_shape) == 1 or (len(tgt_shape) == 2 and tgt_shape[1] == 1):
                target_data = np.random.randint(0, pred_shape[-1], tgt_shape)
                torch_target = torch.tensor(target_data, device=device, dtype=torch.long)
            else:
                torch_target = create_tensor({'shape': tgt_shape, 'dtype': torch.float32, 
                                           'initializer': 'random', 'requires_grad': False}, 
                                          device, 'torch')
        else:
            # Standard target creation
            if classification_target:
                # Classification target
                batch_size = pred_shape[0]
                num_classes = pred_shape[1] if len(pred_shape) > 1 else 2
                target_data = np.random.randint(0, num_classes, (batch_size,))
                torch_target = torch.tensor(target_data, device=device, dtype=torch.long)
            else:
                # Regression target
                torch_target = create_tensor({'shape': tgt_shape, 'dtype': torch.float32,
                                           'initializer': 'random', 'requires_grad': False},
                                          device, 'torch')
        
        torch_inputs.append(torch_target)
    
    # Add module reference if needed
    if module_func:
        if nn_functional:
            torch_inputs.insert(0, torch.nn.functional)
        else:
            torch_inputs.insert(0, torch)
    
    # Run torch forward
    torch_out, torch_fw_time = time_fun(function, *torch_inputs)
    if not isinstance(torch_out, torch.Tensor):
        torch_out = torch_out[0]
    
    # Run torch backward
    if backward:
        _, torch_bw_time = time_fun(torch_out.backward, torch.ones_like(torch_out))
    
    # Prepare inputs for synapx
    torch_inputs = torch_inputs[1:] if module_func else torch_inputs
    
    # Create synapx inputs
    if input_callback:
        syn_inputs = input_callback(inputs, device, 'synapx')
        if loss_function:
            # Convert torch target to synapx
            if not torch_target.is_floating_point():
                syn_target = synapx.tensor(torch_target.detach(), 
                                         dtype=torch_target.dtype, device=device)
            else:
                syn_target = synapx.tensor(torch_target.detach(), 
                                         requires_grad=torch_target.requires_grad,
                                         dtype=torch_target.dtype, device=device)
            syn_inputs.append(syn_target)
    else:
        syn_inputs = []
        for i, inp in enumerate(torch_inputs):
            if loss_function and i == len(torch_inputs) - 1:  # Target tensor
                if not inp.is_floating_point():
                    syn_inputs.append(synapx.tensor(inp.detach(), 
                                                  dtype=inp.dtype, device=device))
                else:
                    syn_inputs.append(synapx.tensor(inp.detach(), requires_grad=inp.requires_grad,
                                                  dtype=inp.dtype, device=device))
            else:
                syn_inputs.append(synapx.tensor(inp.detach(), requires_grad=inp.requires_grad,
                                              dtype=inp.dtype, device=device))
    
    # Add module reference for synapx
    if module_func:
        if nn_functional:
            syn_inputs.insert(0, synapx.nn.functional)
        else:
            syn_inputs.insert(0, synapx)
    
    # Run synapx forward
    syn_out, syn_fw_time = time_fun(function, *syn_inputs)
    if not isinstance(syn_out, synapx.Tensor):
        syn_out = syn_out[0]
    
    # Run synapx backward
    if backward:
        _, syn_bw_time = time_fun(syn_out.backward, synapx.ones_like(syn_out))
    
    # Remove module reference for gradient checking
    syn_inputs = syn_inputs[1:] if module_func else syn_inputs
    
    # Print timing results
    if backward:
        print(f'\n{name}, device: {device}, torch/synapx ' +
              f'fp: {torch_fw_time*1000:.2f} / {syn_fw_time*1000:.2f} ms, ' +
              f'bp: {torch_bw_time*1000:.2f} / {syn_bw_time*1000:.2f} ms')
    else:
        print(f'\n{name}, device: {device}, torch/synapx ' +
              f'fp: {torch_fw_time*1000:.2f} / {syn_fw_time*1000:.2f} ms')
    
    # Check output
    assert check_tensors(syn_out, torch_out, atol=atol, rtol=rtol)
    
    # Check gradients
    if backward:
        for synap_inp, torch_inp in zip(syn_inputs, torch_inputs):
            if torch_inp.requires_grad:
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

    
# *************************
# ******* Other ops *******
# *************************

def test_clone():
    op_tester([(1000, 1000)], lambda x: x.clone(), name='clone')

def test_addmm():
    op_tester([(10, 10), (10, 10), (10, 10)], lambda engine, x, y, z: engine.addmm(x, y, z), name='addmm', module_func=True)

def test_log():
    op_tester([{'shape': (1000, 1000), 'initializer': lambda s: np.random.rand(*s) * 5 + 0.1}], 
              lambda x: x.log(), name='log')

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
    op_tester([{'shape': (4, 4)}], lambda x: x.max(), name='max')
    op_tester([{'shape': (4, 4), 'initializer': 'ones'}], lambda x: x.max(), name='max_ones')
    op_tester([{'shape': (1000, 1500)}], lambda x: x.max(dim=0), name='max_d0')
    op_tester([{'shape': (1000, 1500), 'initializer': 'ones'}], lambda x: x.max(dim=0), name='max_d0_ones')
    op_tester([{'shape': (1000, 1500)}], lambda x: x.max(dim=1), name='max_d1')
    op_tester([{'shape': (1000, 1500), 'initializer': 'ones'}], lambda x: x.max(dim=1), name='max_d1_ones')
    op_tester([{'shape': (1000, 1500)}], lambda x: x.max(dim=1, keepdim=True), name='max_d1')
    op_tester([{'shape': (1000, 3, 4, 5)}], lambda x: x.max(dim=2), name='max_d2')
    op_tester([{'shape': (1000, 3, 4, 5)}], lambda x: x.max(dim=3), name='max_d3')
    op_tester([{'shape': (1000, 3, 4, 5)}], lambda x: x.max(dim=-1), name='max_d3')
    op_tester([{'shape': (1000, 3, 4, 5)}], lambda x: x.max(dim=-2), name='max_d3')

def test_min():
    op_tester([{'shape': (4, 4)}], lambda x: x.min(), name='min')
    op_tester([{'shape': (4, 4), 'initializer': 'ones'}], lambda x: x.min(), name='min_ones')
    op_tester([{'shape': (1000, 1500)}], lambda x: x.min(dim=0), name='min_d0')
    op_tester([{'shape': (1000, 1500), 'initializer': 'ones'}], lambda x: x.min(dim=0), name='min_d0_ones')
    op_tester([{'shape': (1000, 1500)}], lambda x: x.min(dim=1), name='min_d1')
    op_tester([{'shape': (1000, 1500), 'initializer': 'ones'}], lambda x: x.min(dim=1), name='min_d1_ones')
    op_tester([{'shape': (1000, 1500)}], lambda x: x.min(dim=1, keepdim=True), name='min_d1')
    op_tester([{'shape': (1000, 3, 4, 5)}], lambda x: x.min(dim=2), name='min_d2')
    op_tester([{'shape': (1000, 3, 4, 5)}], lambda x: x.min(dim=3), name='min_d3')
    op_tester([{'shape': (1000, 3, 4, 5)}], lambda x: x.min(dim=-1), name='min_d3')
    op_tester([{'shape': (1000, 3, 4, 5)}], lambda x: x.min(dim=-2), name='min_d3')

def test_squeeze():
    op_tester([(100, 1)], lambda x: x.squeeze(dim=1), name='squeeze')
    op_tester([(100, 1)], lambda x: x.squeeze(dim=0), name='squeeze_no_effect')
    op_tester([(1, 3)], lambda x: x.squeeze(dim=0), name='squeeze')
    op_tester([(1, 100, 1, 2)], lambda x: x.squeeze(dim=(0, -2)), name='squeeze')
    op_tester([(1, 100, 1, 2)], lambda x: x.squeeze(), name='squeeze')
    
def test_unsqueeze():
    op_tester([(1000,)], lambda x: x.unsqueeze(dim=1), name='unsqueeze')
    op_tester([(1000, 23)], lambda x: x.unsqueeze(dim=-2), name='unsqueeze')

def test_reshape():
    op_tester([(100000,)], lambda x: x.reshape((1000, 100)), name='reshape')
    op_tester([(10000, 200)], lambda x: x.reshape((-1, 50)), name='reshape')
    
def test_movedim():
    op_tester([(100, 200, 300)], lambda x: x.movedim(0, 1), name='movedim')
    op_tester([(10000, 200)], lambda x: x.movedim(-1, -2), name='movedim')
    
def test_transpose():
    op_tester([(100, 200, 300)], lambda x: x.transpose(2, 1), name='transpose')
    op_tester([(100, 200, 300)], lambda x: x.transpose(0, 1), name='transpose')
    
def test_slice():
    op_tester([(30, 40, 20, 10)], lambda x: x[10:14, 3:, :, :], name='slice')
    op_tester([(30, 40, 20, 10)], lambda x: x[10:25, :5, :12, 4:], name='slice')
    op_tester([(30, 40, 20, 10)], lambda x: x[10:11, 4:8, :12, 0:-1], name='slice')


# **********************************
# ******* Array manipulation *******
# **********************************

def test_stack():
    op_tester([(100,), (100,), (100,)], lambda engine, *x: engine.stack(tuple(x), dim=0), name='stack', module_func=True)
    op_tester([(100, 100), (100, 100), (100, 100)], lambda engine, *x: engine.stack(x), name='stack', module_func=True)
    op_tester([(100, 100), (100, 100), (100, 100)], lambda engine, *x: engine.stack(x, dim=1), name='stack', module_func=True)
    op_tester([(100, 100, 100), (100, 100, 100), (100, 100, 100)], lambda engine, *x: engine.stack(x, dim=2), name='stack',module_func=True)

def test_concat():
    op_tester([(100, 200, 4), (100, 200, 3)], lambda engine, *x: engine.concat(tuple(x), dim=-1), name='concat', module_func=True)
    op_tester([(50, 200, 4), (100, 200, 4)], lambda engine, *x: engine.concat(x), name='concat', module_func=True)
    op_tester([(100, 200, 4), (22, 200, 4)], lambda engine, *x: engine.concat(list(x), dim=0), name='concat', module_func=True)

def test_unbind():
    op_tester([(100, 200, 300)], lambda engine, x: engine.unbind(x, dim=-1), name='unbind', module_func=True)
    op_tester([(100, 200, 300)], lambda engine, x: engine.unbind(x), name='unbind', module_func=True)
    op_tester([(100, 200, 300, 2)], lambda engine, x: engine.unbind(x, dim=1), name='unbind', module_func=True)

# ***************************
# ******* Module ops ********
# ***************************

def test_linear():
    op_tester([(100, 200), (300, 200)], lambda F, x, w: F.linear(x, w), name='linear', module_func=True, nn_functional=True)
    op_tester([(100, 200), (300, 200)], lambda F, x, w: F.linear(x, w), name='linear', module_func=True, nn_functional=True)
    op_tester([(100, 200), (300, 200), (100, 300)], lambda F, x, w, b: F.linear(x, w, b), name='linear', module_func=True, nn_functional=True)

def test_flatten():
    op_tester([(100, 200, 300)], lambda x: x.flatten(), name='flatten')
    op_tester([(100, 200, 300)], lambda x: x.flatten(start_dim=0, end_dim=1), name='flatten_0_1')
    op_tester([(100, 200, 300)], lambda x: x.flatten(start_dim=1, end_dim=2), name='flatten_1_2')
    op_tester([(100, 200, 300)], lambda x: x.flatten(start_dim=0, end_dim=2), name='flatten_0_2')
    op_tester([(100, 200, 300)], lambda x: x.flatten(start_dim=0, end_dim=0), name='flatten_0_0')


# === Activations

def test_relu():
    op_tester([(100, 200, 300)], lambda x: x.relu(), name='relu')
    
def test_sigmoid():
    op_tester([(100, 200, 300)], lambda x: x.sigmoid(), name='sigmoid')
    
def test_softmax():
    op_tester([(100, 200)], lambda x: x.softmax(dim=1), name='softmax')

def test_softmax():
    op_tester([(100, 200)], lambda x: x.log_softmax(dim=1), name='log_softmax')

# === Losses

def test_mse_loss():
    op_tester(
        [{'shape': (100, 100), 'requires_grad': True}], 
        lambda F, inp, target: F.mse_loss(inp, target, reduction='mean'), 
        name='mse_loss', module_func=True, nn_functional=True, loss_function=True
    )

def test_nll_loss():
    op_tester(
        [{'shape': (100, 10), 'initializer': 'randn', 'requires_grad': True}],
        lambda F, inp, target: F.nll_loss(F.log_softmax(inp, dim=1), target, reduction='mean'),
        name='nll_loss', 
        module_func=True, 
        nn_functional=True, 
        loss_function=True,
        classification_target=True
    )