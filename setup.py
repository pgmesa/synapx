import os
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


cwd = Path(__file__).parent.absolute()

synapx_lib_dir = cwd / 'synapx/lib'
lib_dirs = [str((synapx_lib_dir/d).relative_to(cwd)) for d in os.listdir(synapx_lib_dir)]


class SkipBuildExt(build_ext):
    
    def build_extension(self, ext):
        # Skip building the extension entirely
        print(f"Skipping build for extension {ext.name}")


extensions = [
    Extension(
        'synapx._C',
        sources=[],
        libraries=['synapx'],
        language="C++",
        library_dirs=lib_dirs
    )
]

setup(
    ext_modules=extensions,
    cmdclass={'build_ext': SkipBuildExt},
)