
import os
import site
from pathlib import Path

env_packages_path = site.getsitepackages()
package_path = Path(__file__).parent.absolute()

# Add synapx dlls
synapx_dll_dir = package_path/'lib'
os.add_dll_directory(str(synapx_dll_dir))

# import torch  # TODO: Ensures torch's shared libraries are loaded
# Temporal:: Manually add torch dlls
torch_dll_dir = package_path / '../libsynapx/external/libtorch/lib'
os.add_dll_directory(str(torch_dll_dir))

from synapx._C import *