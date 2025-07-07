
import os
from pathlib import Path
from setuptools import setup, Extension


cwd = Path(__file__).parent.absolute()

synapx_lib_dir = cwd / 'synapx/lib'
lib_dirs = [str((synapx_lib_dir/d).relative_to(cwd)) for d in  os.listdir(synapx_lib_dir)]

extensions = [
    Extension(
        'synapx._C',
        libraries=['synapx'],
        language="C++",
        sources=[],
        library_dirs=lib_dirs
    )
]

setup(
    ext_modules=extensions,
)