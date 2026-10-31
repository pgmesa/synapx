
import pytest
import torch
import synapx

from utils import check_tensors


def check_grads(grad, grad_t):
    # A gradient sums many terms, so an element that almost cancels keeps the
    # rounding error of the largest ones: compare against the tensor's scale
    return check_tensors(grad, grad_t, atol=1e-5 * grad_t.abs().max().item())


def copy_to_torch(param):
    return torch.nn.Parameter(param.torch().clone())


@pytest.mark.parametrize("stride, padding, dilation, bias", [
    (1, 0, 1, True),
    (1, 1, 1, False),
    (2, 1, 1, True),
    (1, 2, 2, True),
    ((2, 1), (0, 1), (1, 2), True),
])
def test_conv2d(stride, padding, dilation, bias):
    data = torch.randn(4, 3, 11, 9)

    conv = synapx.nn.Conv2d(3, 5, 3, stride, padding, dilation, bias)
    conv_t = torch.nn.Conv2d(3, 5, 3, stride, padding, dilation, bias=bias)
    conv_t.weight = copy_to_torch(conv.weight)
    if bias: conv_t.bias = copy_to_torch(conv.bias)

    inp = synapx.tensor(data, requires_grad=True)
    out = conv(inp)
    (out * out).sum().backward()

    inp_t = data.clone().requires_grad_()
    out_t = conv_t(inp_t)
    (out_t * out_t).sum().backward()

    assert check_tensors(out, out_t)
    assert check_grads(inp.grad, inp_t.grad)
    assert check_grads(conv.weight.grad, conv_t.weight.grad)
    if bias: assert check_grads(conv.bias.grad, conv_t.bias.grad)


def test_conv2d_padding_same():
    data = torch.randn(2, 1, 8, 8)
    conv = synapx.nn.Conv2d(1, 2, 3, padding='same', bias=False)
    assert tuple(conv(synapx.tensor(data)).shape) == (2, 2, 8, 8)


@pytest.mark.parametrize("kernel_size, stride, padding, dilation", [
    (2, None, 0, 1),
    (3, 2, 1, 1),
    (3, 1, 0, 2),
    ((3, 2), (2, 1), (1, 0), 1),
])
def test_max_pool2d(kernel_size, stride, padding, dilation):
    data = torch.randn(4, 3, 12, 10)

    inp = synapx.tensor(data, requires_grad=True)
    out = synapx.nn.MaxPool2d(kernel_size, stride, padding, dilation)(inp)
    (out * out).sum().backward()

    inp_t = data.clone().requires_grad_()
    out_t = torch.nn.MaxPool2d(kernel_size, stride, padding, dilation)(inp_t)
    (out_t * out_t).sum().backward()

    assert check_tensors(out, out_t)
    assert check_grads(inp.grad, inp_t.grad)


@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("track_running_stats", [True, False])
def test_batch_norm2d(affine, track_running_stats):
    bn = synapx.nn.BatchNorm2d(3, affine=affine, track_running_stats=track_running_stats)
    bn_t = torch.nn.BatchNorm2d(3, affine=affine, track_running_stats=track_running_stats)

    def step(data):
        inp = synapx.tensor(data, requires_grad=True)
        out = bn(inp)
        (out * out * out).sum().backward()

        inp_t = data.clone().requires_grad_()
        out_t = bn_t(inp_t)
        (out_t * out_t * out_t).sum().backward()

        assert check_tensors(out, out_t)
        assert check_grads(inp.grad, inp_t.grad)

    for _ in range(3):
        step(torch.randn(8, 3, 5, 5) * 2 + 1)

    if affine:
        assert check_grads(bn.weight.grad, bn_t.weight.grad)
        assert check_grads(bn.bias.grad, bn_t.bias.grad)
    if track_running_stats:
        assert check_tensors(bn.running_mean, bn_t.running_mean)
        assert check_tensors(bn.running_var, bn_t.running_var)

    bn.eval()
    bn_t.eval()
    step(torch.randn(8, 3, 5, 5) * 2 + 1)


def test_cnn_train_step_matches_torch():
    # dropout out, so both libraries run exactly the same computation
    def block(nn, c_in, c_out):
        return [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2)]

    def model(nn):
        return nn.Sequential(*block(nn, 1, 4), *block(nn, 4, 8), nn.Flatten(), nn.Linear(8 * 7 * 7, 10))

    net, net_t = model(synapx.nn), model(torch.nn)
    for p, p_t in zip(net.parameters(), net_t.parameters()):
        p_t.data.copy_(p.torch())

    # SGD: Adam divides by the size of the gradient, which amplifies the rounding
    # error of the gradients that are almost zero
    opt = synapx.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    opt_t = torch.optim.SGD(net_t.parameters(), lr=0.01, momentum=0.9)
    target = torch.randint(0, 10, (16,))

    for _ in range(3):
        data = torch.randn(16, 1, 28, 28)

        loss = synapx.nn.CrossEntropyLoss()(net(synapx.tensor(data)), synapx.tensor(target, dtype=torch.long))
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_t = torch.nn.CrossEntropyLoss()(net_t(data), target)
        opt_t.zero_grad()
        loss_t.backward()
        opt_t.step()

        assert check_tensors(loss, loss_t)

    for p, p_t in zip(net.parameters(), net_t.parameters()):
        # After a few steps rounding can flip a value across a kink of ReLU or
        # MaxPool, which moves that parameter by up to ~3e-4
        assert check_tensors(p, p_t, atol=2e-3, rtol=1e-3)
