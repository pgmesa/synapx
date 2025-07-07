
import os
import sys
import ctypes
import platform
import importlib.util
from pathlib import Path


__version__ = "0.1.0"

package_path = Path(__file__).parent.absolute()

# Add synapx dlls
synapx_lib_dirs = package_path / 'lib'
libtorch_supported_versions = {
    f.split('-')[1][:-2]: f for f in os.listdir(synapx_lib_dirs) if os.path.isdir(synapx_lib_dirs / f)
}

def print_supported_versions():
    print(f"\nThis SynapX version ({__version__}) supports:")
    for v in libtorch_supported_versions:
        print(f"- torch {v}.X")
    print()

# Ensures libtorch shared libraries are loaded
try:
    import torch
    torch_lib_dir = Path(torch.__file__).parent / 'lib' 
except Exception as e:
    print("\n[x] Could not load 'torch' module")
    print("SynapX requires LibTorch compiled shared libraries to be installed and available in the environment.")
    print("Please ensure you have a supported PyTorch version installed.")
    print_supported_versions()
    print("\nFor installation instructions, visit the official PyTorch website: https://pytorch.org/")
    print(f"Error details: {e}")
    raise

torch_version = '.'.join(torch.__version__.split('.')[:2])

if torch_version in libtorch_supported_versions:
    target_synapx_lib_dir = synapx_lib_dirs / libtorch_supported_versions[torch_version]
    synapx_lib_dir = target_synapx_lib_dir
else:
    print(f"\n[x] Current installed torch version ({torch.__version__}) is not supported")
    print_supported_versions()
    raise RuntimeError(f"Not supported torch version ({torch.__version__})")

# Platform-specific shared library loading
if platform.system() == 'Windows':
    os.add_dll_directory(str(target_synapx_lib_dir))
    extension = '.pyd'
else:
    # Load core library first
    core_lib_path = target_synapx_lib_dir / 'libsynapx.so'
    ctypes.CDLL(str(core_lib_path), mode=ctypes.RTLD_GLOBAL)
    extension = '.so'

# Dynamically identify and load the `_C` module
_C_module_file = None
for file in target_synapx_lib_dir.iterdir():
    if file.suffix == extension and file.stem.startswith('_C'):
        _C_module_file = file
        break

if not _C_module_file:
    raise ImportError("Cannot find the synapx._C shared library file for " + 
                      f"libtorch {torch_version} in {target_synapx_lib_dir}")

# Load the C++ extension as a Python module
spec = importlib.util.spec_from_file_location("synapx._C", _C_module_file)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load spec for {_C_module_file}")

_C = importlib.util.module_from_spec(spec)
sys.modules["synapx._C"] = _C
spec.loader.exec_module(_C)

# Expose everything from the dynamically loaded _C module
from synapx._C import *

# Import commonly used subpackages
from synapx import (
    nn as nn,
    optim as optim
)
