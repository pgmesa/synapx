
import os
import platform
from pathlib import Path

package_path = Path(__file__).parent.absolute()

# Add synapx dlls
synapx_libs_dir = str(package_path/'lib')

if platform.system() == 'Windows':
    os.add_dll_directory(synapx_libs_dir)
else:
    os.environ['LD_LIBRARY_PATH'] = f'{synapx_libs_dir}:' + os.environ.get('LD_LIBRARY_PATH', '')

# Ensures torch shared libraries are loaded
import torch

from synapx._C import *