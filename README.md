# SynapX

[![PyPI version](https://badge.fury.io/py/synapx.svg)](https://badge.fury.io/py/synapx)
[![Python versions](https://img.shields.io/pypi/pyversions/synapx.svg)](https://pypi.org/project/synapx/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## What is this project?

**SynapX** is a deep learning library that implements its core functionality (autograd and tensor operations) in C++ and exposes it to Python via bindings. The goal was to leverage the computational power of C++ while keeping Python's ease of use. This project uses **libtorch** as its main dependency to serve as the backend for tensor operations across multiple devices (CPU, CUDA, ROCm, etc).

The project is designed to be compatible with Windows and Linux (tested on Windows 10 and Ubuntu 22.04) and implements a PyTorch-like API to make it familiar and easy to use - if you know PyTorch, you know SynapX.

## Why I built this

The aim was to create an autograd engine in C++, implement Python bindings, build a Deep Learning library on top of it, and package everything into a cross-platform Python package. It's essentially an exploration of how automatic differentiation works under the hood, combined with the practical challenge of bridging C++ performance with Python usability.

Any contributions or ideas are more than welcome!

> **Note:** This project builds on my previous exploration [synapgrad](https://github.com/pgmesa/synapgrad), which implemented the same autograd concepts purely in Python using numpy for tensor operations.

## Quick Start

```python
import synapx as sx

W = sx.randn(3, 4, requires_grad=True)
X = sx.randn(2, 3, requires_grad=True)
b = sx.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)

# Matrix multiplication and broadcasting (addmm or linear could also be used here)
Y = sx.matmul(X, W) + b  # Shape: (2, 4) 

# Slice operations
Y_slice = Y[:, 1:3]  # Take columns 1-2

# Unbind along dimension 0 (split into individual tensors)
y1, y2 = sx.unbind(Y_slice, dim=0)

# Compute loss
loss = (y1 * y2).sum()

# Gradients are computed automatically
loss.backward()

print(f"W.grad shape: {W.grad.shape}")
print(f"X.grad shape: {X.grad.shape}")
print(f"b.grad: {b.grad}")

# Use no_grad context for inference
with sx.no_grad():
    inference_result = sx.matmul(X, W) + b
    print(f"Inference (no gradients): {inference_result}")
```

Simple neural network:
```python
import synapx
import synapx.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 128)
        self.linear2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)

model = SimpleNet()
x = synapx.randn(32, 784)
y = model(x)
```

## Installation

```bash
pip install synapx
```

The package automatically detects your PyTorch installation and uses its attached libtorch version as the backend. Make sure you have PyTorch installed:

```bash
pip install torch
```

## Project Structure

The project is organized into 4 main parts:

### libsynapx/
This is the C++ library that powers everything. Here's where the critical components live:
- **Tensor class**: A wrapper around libtorch tensors
- **Autograd engine**: Implements backward functions, backpropagation algorithm, and graph components
- **Dynamic computation graph**: Builds the DAG as operations are chained

**Dependencies**: The main dependencies are `pybind11`, `libtorch`, spdlog, and VCPKG. Both pybind11 and libtorch can be installed with pip and will be automatically detected during the CMake build process. SynapX links to the shared libraries of libtorch from your Python torch installation.

**Compatibility note**: Building SynapX against libtorch+cpu is sufficient to work with both CPU-only and CUDA versions. Compiling against the CPU-only version provides support for both, but not the other way around.

### synapx/
Once the C++ code is compiled and installed, this directory contains the complete Python package with all components linked together: the Deep Learning library, tensor operations, and autograd engine.

The package is compiled for each libtorch minor version it supports. A SynapX compiled with libtorch 2.3.X won't necessarily work with 2.4.X, and the same goes for different Python versions. The correct C++ library is loaded dynamically at runtime by checking your installed PyTorch version.

### examples/
Contains practical examples showcasing the library in action. You'll need additional dependencies for these:
```bash
pip install scikit-learn numpy matplotlib pkbar
```

The examples include simple problems like make_moons and MNIST classification. You can easily switch between PyTorch and SynapX backends to compare performance and functionality.

**Performance note**: Currently, there's still a significant performance gap between PyTorch and SynapX, even though both use the same underlying libtorch operations. This is an early version where I focused on getting everything working correctly - the optimization phase comes next, and there are many known bottlenecks and excessive memory consumption issues that can be addressed.

### tests/
Contains tests for almost every tensor operation supported in SynapX, as well as tests for layers, activations, and other components.

Run all tests:
```bash
pip install pytest
python -m pytest
```

To compare PyTorch and SynapX performance for each implemented operation:
```bash
python -m pytest ./tests/test_ops.py -s
```

## Building from Source

### Setting up CMake presets

Create a `CMakeUserPresets.json` file in the `libsynapx/` directory:

**Windows example:**
```json
{
  "version": 3,
  "configurePresets": [
    {
      "name": "vs2022-release",
      "inherits": "windows-release",
      "displayName": "Visual Studio 2022 Windows Release",
      "generator": "Visual Studio 17 2022",
      "environment": {
        "VCPKG_ROOT": "C:\\Users\\<user>\\vcpkg"
      },
      "cacheVariables": {
        "VCPKG_TARGET_TRIPLET": "x64-windows",
        "CMAKE_CXX_COMPILER": "cl",
        "BUILD_PYTHON_BINDINGS": "ON",
        "BUILD_EXAMPLES": "OFF"
      }
    }
  ],
  "buildPresets": [
    {
      "name": "vs2022-release",
      "configurePreset": "vs2022-release",
      "displayName": "Visual Studio 2022 Windows Release Build",
      "configuration": "Release"
    }
  ]
}
```

**Linux example:**
```json
{
  "version": 3,
  "configurePresets": [
    {
      "name": "ninja-release",
      "inherits": "linux-release",
      "displayName": "Ninja g++ Release",
      "generator": "Ninja",
      "environment": {
        "VCPKG_ROOT": "/home/<user>/vcpkg"
      },
      "cacheVariables": {
        "VCPKG_TARGET_TRIPLET": "x64-linux",
        "CMAKE_C_COMPILER": "/usr/bin/gcc-12",
        "CMAKE_CXX_COMPILER": "/usr/bin/g++-12",
        "CMAKE_CUDA_HOST_COMPILER": "/usr/bin/gcc-12",
        "BUILD_PYTHON_BINDINGS": "ON",
        "BUILD_EXAMPLES": "OFF"
      }
    }
  ]
}
```

**Note**: Explicitly specify `CMAKE_CUDA_HOST_COMPILER` when building against libtorch+cuda libraries.

### Building

```bash
cd libsynapx

# Windows
make rebuild preset=vs2022-release target=install

# Linux  
make rebuild preset=ninja-release target=install
```

### Generating Python stubs

After compilation, generate stub files using:
```bash
pip install pybind11-stubgen
python scripts/generate_pyi.py
```

## Supported Versions

SynapX is compiled for specific combinations of Python and libtorch versions. With each release, GitHub Actions automatically builds the necessary wheels and uploads them to PyPI. Currently supported versions:

| Python Version | PyTorch Versions | Status |
|----------------|------------------|--------|
| 3.9           | 2.0.X, 2.1.X, 2.2.X, 2.3.X, 2.4.X, 2.5.X | ✅ |
| 3.10          | 2.0.X, 2.1.X, 2.2.X, 2.3.X, 2.4.X, 2.5.X | ✅ |
| 3.11          | 2.0.X, 2.1.X, 2.2.X, 2.3.X, 2.4.X, 2.5.X | ✅ |
| 3.12          | 2.2.X, 2.3.X, 2.4.X, 2.5.X | ✅ |


## Current Limitations and TODO

### Things that need work and will probably be implemented in upcoming versions:
- **Improve autograd speed**: There are significant bottlenecks that can be optimized
- **Add tensor hooks**: For inspecting or modifying the backward pass from Python. The `retain_grad()` function doesn't work yet because it should be implemented as a tensor hook
- **Fix `__iter__` in Python tensor class**: Current implementation is not optimal, but works.
- **Add remaining conv functionality**: 1D and 2D convolutions, maxpool, avgpool
- **Add CNN MNIST example**

### Future ideas (contributions welcome!):
- **Visualize computation graphs** with graphviz (forward and backward)
- **Add PyNode support** to let users define backward functions in Python that get called from the C++ engine
- **Multi-backend support**: Instead of relying only on libtorch, add the ability to switch between libtorch, xtensor, etc. This would require decoupling some logic from the current Tensor class and restructuring parts of the codebase, but it could be interesting for comparing the autograd engine with different tensor operation backends

The last point would be quite ambitious - adding support for the wide range of operations needed for a complete autograd system using libraries not specifically designed for N-dimensional tensors with complex indexing and slicing would be a significant undertaking.