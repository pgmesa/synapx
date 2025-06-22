
import torch
import synapx

from utils import check_tensors


def check_activation(act, act_t):
    l1 = [[-1.0,-2.0, 4.0, 5.0, 1.0, 7.0], [-1.0,-2.0, 2.0, 1.0, 10.0, -4.0]]
    
    # synapgrad
    a = synapx.tensor(l1, requires_grad=True)
    b = (a*4)/7 - 2
    out = act(b)*b
    out = out.sum()
    out.backward()
    
    # torch
    a_t = torch.tensor(l1, requires_grad=True)
    b_t = (a_t*4)/7 - 2
    out_t = act_t(b_t)*b_t
    out_t = out_t.sum()
    out_t.backward()
    
    print(a.grad)
    print(a_t.grad)
    
    assert check_tensors(out, out_t)
    assert check_tensors(a.grad, a_t.grad)


def test_relu():
    check_activation(synapx.nn.ReLU(), torch.nn.ReLU())
    
    
def test_sigmoid():
    check_activation(synapx.nn.Sigmoid(), torch.nn.Sigmoid())
    
    
# def test_softmax():
#     check_activation(synapx.nn.Softmax(dim=1), torch.nn.Softmax(dim=1))
    
    
# def test_log_softmax():
#     check_activation(synapx.nn.LogSoftmax(dim=1), torch.nn.LogSoftmax(dim=1))