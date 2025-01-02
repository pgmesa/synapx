
import os
import platform
from pathlib import Path


__version__ = "0.1.0"

package_path = Path(__file__).parent.absolute()

# Add synapx dlls
synapx_lib_dir = package_path/'lib'
libtorch_supported_versions = {f.split('-')[1][:-2]:f for f in os.listdir(synapx_lib_dir)}

# Ensures libtorch shared libraries are loaded
try:
    import torch
except Exception as e:
    print("[x] Could not load 'torch' module")
    print("SynapX requires LibTorch compiled shared libraries to be installed and available in the environment.")
    print("Please ensure you have a supported PyTorch version installed.")
    print()
    print(f"This SynapX version ({__version__}) supports:")
    for v in libtorch_supported_versions:
        print(f"- torch {v}.x")
    print("\nFor installation instructions, visit the official PyTorch website: https://pytorch.org/")
    print(f"Error details: {e}")
    raise

torch_version = '.'.join(torch.__version__.split('.')[:2])
target_synapx_lib_dir = str(synapx_lib_dir / libtorch_supported_versions[torch_version])

if platform.system() == 'Windows':
    os.add_dll_directory(target_synapx_lib_dir)
else:
    os.environ['LD_LIBRARY_PATH'] = f'{target_synapx_lib_dir}:' + os.environ.get('LD_LIBRARY_PATH', '')


from synapx._C import *