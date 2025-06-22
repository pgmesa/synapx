
import torch
import synapx

from utils import check_tensors

    
def test_linear():
    for i in range(2):
        data = torch.rand((10,6), dtype=torch.float32)
        
        bias = True
        if i == 0: bias = False
        
        # synapgrad
        inp = synapx.tensor(data, requires_grad=True)
        linear = synapx.nn.Linear(6,3, bias=bias)
        out = linear(inp)
        out = out.sum()
        out.backward()

        # torch
        inp_t = data.detach().requires_grad_(True)
        linear_t = torch.nn.Linear(6,3, bias=bias)
        linear_t.weight = torch.nn.parameter.Parameter(linear.weight.torch())
        if bias:
            linear_t.bias = torch.nn.parameter.Parameter(linear.bias.torch())
        out_t = linear_t(inp_t)
        out_t = out_t.sum()
        out_t.backward()

        params = linear.parameters()
        params_t = list(linear_t.parameters())

        assert len(params) == len(params_t)
        for p, p_t in zip(params, params_t):
            assert check_tensors(p, p_t)
            assert check_tensors(p.grad, p_t.grad)
            
        assert check_tensors(out, out_t)
        assert check_tensors(inp.grad, inp_t.grad)
        
    
def test_flatten():
    data = torch.randn((30,28,28,3,4))
    
    # synapgrad
    inp = synapx.tensor(data, requires_grad=True)
    linear = synapx.nn.Flatten(start_dim=1, end_dim=2)
    out_l = linear(inp)
    out = out_l.sum()
    out.backward()
    
    # torch
    inp_t = data.requires_grad_(True)
    linear_t = torch.nn.Flatten(start_dim=1, end_dim=2)
    out_tl = linear_t(inp_t)
    out_t = out_tl.sum()
    out_t.backward()

    assert check_tensors(out_l, out_tl)
    assert check_tensors(inp.grad, inp_t.grad)