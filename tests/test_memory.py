import gc

import pytest
import torch
import synapx


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")

# torch's lazy CUDA init resets the caching allocator stats, so it must run
# before synapx allocates anything or memory_allocated() misses those blocks.
if torch.cuda.is_available():
    torch.cuda.init()

# Ops whose backward node saves the forward output. Saving it as a regular
# Tensor creates the cycle output -> grad_fn -> node -> output and the whole
# graph is never freed.
OUTPUT_SAVING_OPS = {
    "exp": lambda x: x.exp(),
    "sqrt": lambda x: (x * x).sqrt(),
    "pow": lambda x: x.pow(3.0),
    "relu": lambda x: x.relu(),
    "sigmoid": lambda x: x.sigmoid(),
    "softmax": lambda x: x.softmax(dim=1),
    "log_softmax": lambda x: x.log_softmax(dim=1),
    "max": lambda x: x.max(),
    "min": lambda x: x.min(),
}


@requires_cuda
@pytest.mark.parametrize("op_name", OUTPUT_SAVING_OPS)
def test_graph_is_freed_after_backward(op_name):
    op = OUTPUT_SAVING_OPS[op_name]
    x = synapx.randn((512, 512), requires_grad=True, device="cuda")

    # Warm-up iteration so x.grad is already allocated
    op(x).sum().backward()
    gc.collect()
    baseline = torch.cuda.memory_allocated()

    for _ in range(5):
        op(x).sum().backward()
    gc.collect()

    leaked = torch.cuda.memory_allocated() - baseline
    assert leaked == 0, f"{op_name}: {leaked} bytes not released after backward"


@requires_cuda
@pytest.mark.parametrize("op_name", OUTPUT_SAVING_OPS)
def test_graph_is_freed_without_backward(op_name):
    op = OUTPUT_SAVING_OPS[op_name]
    x = synapx.randn((512, 512), requires_grad=True, device="cuda")
    gc.collect()
    baseline = torch.cuda.memory_allocated()

    for _ in range(5):
        op(x).sum()
    gc.collect()

    leaked = torch.cuda.memory_allocated() - baseline
    assert leaked == 0, f"{op_name}: {leaked} bytes not released"
