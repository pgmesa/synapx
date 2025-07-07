
import os
import platform
from pathlib import Path
from setuptools import setup, Extension
from wheel.bdist_wheel import bdist_wheel


system = platform.system()
cwd = Path(__file__).parent.absolute()


if system == 'Linux':
    synapx_lib_dir = cwd / 'synapx/lib'
    lib_dirs = [str((synapx_lib_dir/d).relative_to(cwd)) for d in  os.listdir(synapx_lib_dir)]
    print(lib_dirs)

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
    
elif system == 'Windows':
    class BDistWheelCustom(bdist_wheel):
        
        def finalize_options(self):
            bdist_wheel.finalize_options(self)
            self.root_is_pure = False

    setup(
        cmdclass={
            'bdist_wheel': BDistWheelCustom,
        },
    )
    
else:
    raise RuntimeError(f"Current system is not supported '{system}'")