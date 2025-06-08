
import torch
import synapx

from utils import check_tensors


# Check with pytorch that gradients are correct when applying different tensor operations
def test_engine_chain_ops():
    l1 = [[[2.0, 4.0], [2.0,4.3]], [[2.0, 4.0], [2.0,4.3]]]
    l2 = [2.0, 4.0]
    
    # synapgrad
    a = synapx.tensor(l1, requires_grad=True)
    b = synapx.tensor(l2, requires_grad=True)
    c = (a.exp()+b)*b.log().sqrt().mean(dim=0, keepdim=True)
    c.sum().backward()
    
    # torch
    a_t = torch.tensor(l1, requires_grad=True)
    b_t = torch.tensor(l2, requires_grad=True)
    c_t = (a_t.exp()+b_t)*b_t.log().sqrt().mean(dim=0, keepdim=True)
    c_t.sum().backward()
    
    assert check_tensors(c, c_t)
    assert check_tensors(b.grad, b_t.grad)
    
    
def test_engine_reductions_ops():
    # synapx
    a = synapx.rand((4,64,16,16), requires_grad=True, dtype=torch.float32)
    c = a.mean(dim=(0,2,3), keepdim=True).max()
    c.sum().backward()
    
    # torch
    a_t = a.torch()
    a_t.requires_grad_(True);
    c_t = a_t.mean(dim=(0,2,3), keepdim=True).max()
    c_t.sum().backward()
    
    print(a.grad)
    print(a_t.grad)
    
    assert check_tensors(c, c_t)
    assert check_tensors(a.grad, a_t.grad)
    

# def test_engine_shape_manipulation():
#     l1 = [[-4.0, 0.7, 5.0], [6.3, 3.2, 1.3]]
#     l2 = [[2.0, 2,  3.0], [2.4, 1.7, 0.5]]
    
#     # synapgrad
#     a = synapx.tensor(l1, requires_grad=True).unsqueeze(0)**2
#     a.retain_grad()
#     b = 2**synapx.tensor(l2, requires_grad=True).unsqueeze(0)
#     b.retain_grad()
#     c = synapx.tensor(4.0, requires_grad=True)
    
#     out1 = synapx.stack((a.squeeze(), b.squeeze()))[0]
#     out2 = synapx.concat((a*c, b), dim=1).transpose(0, 1)[0, :]
#     out = out1 @ out2.reshape(3).unsqueeze(1)
#     s = out.sum()
#     s.backward()
    
#     ## torch
#     a_t = torch.tensor(l1, requires_grad=True).unsqueeze(0)**2
#     a_t.retain_grad()
#     b_t = 2**torch.tensor(l2, requires_grad=True).unsqueeze(0)
#     b_t.retain_grad()
#     c_t = torch.tensor(4.0, requires_grad=True)
    
#     out1_t = torch.stack((a_t.squeeze(), b_t.squeeze()))[0]
#     out2_t = torch.concat((a_t*c_t, b_t), dim=1).transpose(0, 1)[0, :]
#     out_t = out1_t @ out2_t.reshape(3).unsqueeze(1)
#     s_t = out_t.sum()
#     s_t.backward()

#     assert check_tensors(a, a_t)
#     assert check_tensors(b, b_t)
#     assert check_tensors(c, c_t)
#     assert check_tensors(out, out_t)
#     assert check_tensors(a.grad, a_t.grad)
#     assert check_tensors(b.grad, b_t.grad)
#     assert check_tensors(c.grad, c_t.grad)
    

# def test_engine_multitensor_manipulation():
#     # torch
#     inp_t = torch.randint(0, 10, size=(3, 10), requires_grad=True)
#     unb_t = torch.unbind(inp_t,  dim=0)
#     unb_t = [unb_t[i]*i for i in range(len(unb_t))]
#     stacked_t = torch.stack(unb_t, dim=0) / 2
#     unb2_t = torch.unbind(stacked_t, dim=0)
#     unb2_t = [unb2_t[i]/(i+1) for i in range(len(unb2_t))]
#     concated_t = torch.concat(unb2_t, dim=0)
    
#     concated_t.sum().backward()
    
#     print(concated_t)
#     print(inp_t.grad)
    
#     # synapx
#     inp = synapx.tensor(inp_t, requires_grad=True)
#     unb = synapx.unbind(inp,  dim=0)
#     unb = [unb[i]*i for i in range(len(unb))]
#     stacked = synapx.stack(unb, dim=0) / 2
#     unb2 = synapx.unbind(stacked, dim=0)
#     unb2 = [unb2[i]/(i+1) for i in range(len(unb2))]
#     concated = synapx.concat(unb2, dim=0)
    
#     concated.sum().backward()
    
#     assert check_tensors(concated, concated_t)
#     assert check_tensors(inp.grad, inp_t.grad)