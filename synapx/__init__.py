import os
import sys
import platform
import importlib.util
from pathlib import Path


__version__ = "0.1.0-alpha.1"

package_path = Path(__file__).parent.absolute()

# Add synapx dlls
synapx_lib_dir = package_path / 'lib'
libtorch_supported_versions = {
    f.split('-')[1][:-2]: f for f in os.listdir(synapx_lib_dir) if os.path.isdir(synapx_lib_dir / f)
}

def print_supported_versions():
    print(f"\nThis SynapX version ({__version__}) supports:")
    for v in libtorch_supported_versions:
        print(f"- torch {v}.x")
    print()

# Ensures libtorch shared libraries are loaded
try:
    import torch
except Exception as e:
    print("[x] Could not load 'torch' module")
    print("SynapX requires LibTorch compiled shared libraries to be installed and available in the environment.")
    print("Please ensure you have a supported PyTorch version installed.")
    print_supported_versions()
    print("\nFor installation instructions, visit the official PyTorch website: https://pytorch.org/")
    print(f"Error details: {e}")
    raise

torch_version = '.'.join(torch.__version__.split('.')[:2])

if torch_version in libtorch_supported_versions:
    target_synapx_lib_dir = synapx_lib_dir / libtorch_supported_versions[torch_version]
else:
    print(f"[x] Current installed torch version ({torch_version}.x) is not supported")
    print_supported_versions()
    raise RuntimeError("Not supported torch version")

# Platform-specific shared library loading
if platform.system() == 'Windows':
    os.add_dll_directory(str(target_synapx_lib_dir))
    extension = '.pyd'
else:
    os.environ['LD_LIBRARY_PATH'] = f'{target_synapx_lib_dir}:' + os.environ.get('LD_LIBRARY_PATH', '')
    extension = '.so'

# Dynamically identify and load the `_C` module
_C_module_file = None
for file in target_synapx_lib_dir.iterdir():
    if file.suffix == extension and file.stem.startswith('_C'):
        _C_module_file = file
        break

if not _C_module_file:
    raise ImportError(f"Cannot find the _C shared library file for torch {torch_version} in {target_synapx_lib_dir}")

# Dynamically load the identified `_C` module
spec = importlib.util.spec_from_file_location("synapx._C", _C_module_file)
_C = importlib.util.module_from_spec(spec)
sys.modules["synapx._C"] = _C
spec.loader.exec_module(_C)

# Expose everything from the dynamically loaded _C module
from synapx._C import *
