import sys
import subprocess
from pathlib import Path

import torch
import synapx
import numpy as np


def test_tensor_from_list():
    t = synapx.tensor([[1, 2], [3, 4]])
    assert t.dtype == torch.float32
    assert torch.equal(t.torch(), torch.tensor([[1., 2.], [3., 4.]]))

    t = synapx.tensor([True, False], dtype=torch.bool)
    assert torch.equal(t.torch(), torch.tensor([True, False]))


def test_tensor_from_list_keeps_float64_precision():
    t = synapx.tensor([0.1, 1e-300], dtype=torch.float64)
    assert t.torch().tolist() == [0.1, 1e-300]


def test_tensor_from_numpy():
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    t = synapx.tensor(a, dtype=torch.float64)
    assert t.torch().tolist() == a.tolist()


def test_tensor_without_numpy():
    # Setting sys.modules['numpy'] to None makes `import numpy` fail, as if it
    # were not installed
    code = (
        "import sys; sys.modules['numpy'] = None\n"
        "import torch, synapx\n"
        "t = synapx.tensor([[1, 2], [3, 4]])\n"
        "assert t.torch().tolist() == [[1., 2.], [3., 4.]]\n"
        "t = synapx.tensor(torch.ones(2))\n"
        "assert t.torch().tolist() == [1., 1.]\n"
    )
    root = Path(__file__).parent.parent
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)
