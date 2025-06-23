
import torch
import synapx

from utils import check_tensors


def test_linear_train():
    model = synapx.nn.Linear(3, 3)
    opt = synapx.optim.SGD(model.parameters(), lr=0.1)
    
    model_t = torch.nn.Linear(3, 3)
    model_t.weight = torch.nn.parameter.Parameter(model.weight.torch().clone())
    model_t.bias = torch.nn.parameter.Parameter(model.bias.torch().clone())
    opt_t = torch.optim.SGD(model_t.parameters(), lr=0.1)
    
    for _ in range(100):
        data = torch.randn((3, 3), dtype=torch.float32)
        inp = synapx.tensor(data, requires_grad=True)
        # synapx
        out = model(inp)
        opt.zero_grad()
        out.sum().backward()
        opt.step()

        inp_t = data.requires_grad_(True)
        # torch   
        out_t = model_t(inp_t)
        opt_t.zero_grad()
        out_t.sum().backward()
        opt_t.step()
        
        assert check_tensors(inp, inp_t)
        assert check_tensors(model.weight, model_t.weight)
        assert check_tensors(model.bias, model_t.bias)
        assert check_tensors(out, out_t)
        assert check_tensors(inp.grad, inp_t.grad)
        assert check_tensors(model.weight.grad, model_t.weight.grad)
        assert check_tensors(model.bias.grad, model_t.bias.grad)